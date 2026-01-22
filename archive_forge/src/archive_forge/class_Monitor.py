import argparse
import json
import logging
import os
import signal
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict
from typing import Any, Callable, Dict, Optional, Union
import ray
import ray._private.ray_constants as ray_constants
import ray._private.utils
from ray._private.event.event_logger import get_event_logger
from ray._private.ray_logging import setup_component_logger
from ray._raylet import GcsClient
from ray.autoscaler._private.autoscaler import StandardAutoscaler
from ray.autoscaler._private.commands import teardown_cluster
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.event_summarizer import EventSummarizer
from ray.autoscaler._private.load_metrics import LoadMetrics
from ray.autoscaler._private.prom_metrics import AutoscalerPrometheusMetrics
from ray.autoscaler._private.util import format_readonly_node_type
from ray.core.generated import gcs_pb2
from ray.core.generated.event_pb2 import Event as RayEvent
from ray.experimental.internal_kv import (
class Monitor:
    """Autoscaling monitor.

    This process periodically collects stats from the GCS and triggers
    autoscaler updates.
    """

    def __init__(self, address: str, autoscaling_config: Union[str, Callable[[], Dict[str, Any]]], log_dir: str=None, prefix_cluster_info: bool=False, monitor_ip: Optional[str]=None, retry_on_failure: bool=True):
        self.gcs_address = address
        worker = ray._private.worker.global_worker
        self.gcs_client = GcsClient(address=self.gcs_address)
        if monitor_ip:
            monitor_addr = f'{monitor_ip}:{AUTOSCALER_METRIC_PORT}'
            self.gcs_client.internal_kv_put(b'AutoscalerMetricsAddress', monitor_addr.encode(), True, None)
        _initialize_internal_kv(self.gcs_client)
        if monitor_ip:
            monitor_addr = f'{monitor_ip}:{AUTOSCALER_METRIC_PORT}'
            self.gcs_client.internal_kv_put(b'AutoscalerMetricsAddress', monitor_addr.encode(), True, None)
        self._session_name = self.get_session_name(self.gcs_client)
        logger.info(f'session_name: {self._session_name}')
        worker.mode = 0
        head_node_ip = self.gcs_address.split(':')[0]
        self.load_metrics = LoadMetrics()
        self.last_avail_resources = None
        self.event_summarizer = EventSummarizer()
        self.prefix_cluster_info = prefix_cluster_info
        self.retry_on_failure = retry_on_failure
        self.autoscaling_config = autoscaling_config
        self.autoscaler = None
        self.readonly_config = None
        if log_dir:
            try:
                self.event_logger = get_event_logger(RayEvent.SourceType.AUTOSCALER, log_dir)
            except Exception:
                self.event_logger = None
        else:
            self.event_logger = None
        self.prom_metrics = AutoscalerPrometheusMetrics(session_name=self._session_name)
        if monitor_ip and prometheus_client:
            try:
                logger.info('Starting autoscaler metrics server on port {}'.format(AUTOSCALER_METRIC_PORT))
                kwargs = {'addr': '127.0.0.1'} if head_node_ip == '127.0.0.1' else {}
                prometheus_client.start_http_server(port=AUTOSCALER_METRIC_PORT, registry=self.prom_metrics.registry, **kwargs)
                self.prom_metrics.pending_nodes.clear()
                self.prom_metrics.active_nodes.clear()
            except Exception:
                logger.exception('An exception occurred while starting the metrics server.')
        elif not prometheus_client:
            logger.warning('`prometheus_client` not found, so metrics will not be exported.')
        logger.info('Monitor: Started')

    def _initialize_autoscaler(self):
        if self.autoscaling_config:
            autoscaling_config = self.autoscaling_config
        else:
            self.readonly_config = BASE_READONLY_CONFIG

            def get_latest_readonly_config():
                return self.readonly_config
            autoscaling_config = get_latest_readonly_config
        self.autoscaler = StandardAutoscaler(autoscaling_config, self.load_metrics, self.gcs_client, self._session_name, prefix_cluster_info=self.prefix_cluster_info, event_summarizer=self.event_summarizer, prom_metrics=self.prom_metrics)

    def update_load_metrics(self):
        """Fetches resource usage data from GCS and updates load metrics."""
        response = self.gcs_client.get_all_resource_usage(timeout=60)
        resources_batch_data = response.resource_usage_data
        log_resource_batch_data_if_desired(resources_batch_data)
        if self.readonly_config:
            new_nodes = []
            for msg in list(resources_batch_data.batch):
                node_id = msg.node_id.hex()
                new_nodes.append((node_id, msg.node_manager_address))
            self.autoscaler.provider._set_nodes(new_nodes)
        mirror_node_types = {}
        cluster_full = False
        if hasattr(response, 'cluster_full_of_actors_detected_by_gcs') and response.cluster_full_of_actors_detected_by_gcs:
            cluster_full = True
        for resource_message in resources_batch_data.batch:
            node_id = resource_message.node_id
            if self.readonly_config:
                node_type = format_readonly_node_type(node_id.hex())
                resources = {}
                for k, v in resource_message.resources_total.items():
                    resources[k] = v
                mirror_node_types[node_type] = {'resources': resources, 'node_config': {}, 'max_workers': 1}
            if hasattr(resource_message, 'cluster_full_of_actors_detected') and resource_message.cluster_full_of_actors_detected:
                cluster_full = True
            total_resources = dict(resource_message.resources_total)
            available_resources = dict(resource_message.resources_available)
            waiting_bundles, infeasible_bundles = parse_resource_demands(resources_batch_data.resource_load_by_shape)
            pending_placement_groups = list(resources_batch_data.placement_group_load.placement_group_data)
            use_node_id_as_ip = self.autoscaler is not None and self.autoscaler.config['provider'].get('use_node_id_as_ip', False)
            if use_node_id_as_ip:
                peloton_id = total_resources.get('NODE_ID_AS_RESOURCE')
                if peloton_id is not None:
                    ip = str(int(peloton_id))
                else:
                    ip = node_id.hex()
            else:
                ip = resource_message.node_manager_address
            self.load_metrics.update(ip, node_id, total_resources, available_resources, waiting_bundles, infeasible_bundles, pending_placement_groups, cluster_full)
        if self.readonly_config:
            self.readonly_config['available_node_types'].update(mirror_node_types)

    def get_session_name(self, gcs_client: GcsClient) -> Optional[str]:
        """Obtain the session name from the GCS.

        If the GCS doesn't respond, session name is considered None.
        In this case, the metrics reported from the monitor won't have
        the correct session name.
        """
        if not _internal_kv_initialized():
            return None
        session_name = gcs_client.internal_kv_get(b'session_name', ray_constants.KV_NAMESPACE_SESSION, timeout=10)
        if session_name:
            session_name = session_name.decode()
        return session_name

    def update_resource_requests(self):
        """Fetches resource requests from the internal KV and updates load."""
        if not _internal_kv_initialized():
            return
        data = _internal_kv_get(ray._private.ray_constants.AUTOSCALER_RESOURCE_REQUEST_CHANNEL)
        if data:
            try:
                resource_request = json.loads(data)
                self.load_metrics.set_resource_requests(resource_request)
            except Exception:
                logger.exception('Error parsing resource requests')

    def _run(self):
        """Run the monitor loop."""
        while True:
            try:
                gcs_request_start_time = time.time()
                self.update_load_metrics()
                gcs_request_time = time.time() - gcs_request_start_time
                self.update_resource_requests()
                self.update_event_summary()
                load_metrics_summary = self.load_metrics.summary()
                status = {'gcs_request_time': gcs_request_time, 'time': time.time(), 'monitor_pid': os.getpid()}
                if self.autoscaler and (not self.load_metrics):
                    logger.info('Autoscaler has not yet received load metrics. Waiting.')
                elif self.autoscaler:
                    update_start_time = time.time()
                    self.autoscaler.update()
                    status['autoscaler_update_time'] = time.time() - update_start_time
                    autoscaler_summary = self.autoscaler.summary()
                    try:
                        self.emit_metrics(load_metrics_summary, autoscaler_summary, self.autoscaler.all_node_types)
                    except Exception:
                        logger.exception('Error emitting metrics')
                    if autoscaler_summary:
                        status['autoscaler_report'] = asdict(autoscaler_summary)
                        status['non_terminated_nodes_time'] = self.autoscaler.non_terminated_nodes.non_terminated_nodes_time
                    for msg in self.event_summarizer.summary():
                        for line in msg.split('\n'):
                            logger.info('{}{}'.format(ray_constants.LOG_PREFIX_EVENT_SUMMARY, line))
                            if self.event_logger:
                                self.event_logger.info(line)
                    self.event_summarizer.clear()
                status['load_metrics_report'] = asdict(load_metrics_summary)
                as_json = json.dumps(status)
                if _internal_kv_initialized():
                    _internal_kv_put(ray_constants.DEBUG_AUTOSCALING_STATUS, as_json, overwrite=True)
            except Exception:
                if self.retry_on_failure:
                    logger.exception('Monitor: Execution exception. Trying again...')
                else:
                    raise
            time.sleep(AUTOSCALER_UPDATE_INTERVAL_S)

    def emit_metrics(self, load_metrics_summary, autoscaler_summary, node_types):
        if autoscaler_summary is None:
            return None
        for resource_name in ['CPU', 'GPU', 'TPU']:
            _, total = load_metrics_summary.usage.get(resource_name, (0, 0))
            pending = autoscaler_summary.pending_resources.get(resource_name, 0)
            self.prom_metrics.cluster_resources.labels(resource=resource_name, SessionName=self.prom_metrics.session_name).set(total)
            self.prom_metrics.pending_resources.labels(resource=resource_name, SessionName=self.prom_metrics.session_name).set(pending)
        pending_node_count = Counter()
        for _, node_type, _ in autoscaler_summary.pending_nodes:
            pending_node_count[node_type] += 1
        for node_type, count in autoscaler_summary.pending_launches.items():
            pending_node_count[node_type] += count
        for node_type in node_types:
            count = pending_node_count[node_type]
            self.prom_metrics.pending_nodes.labels(SessionName=self.prom_metrics.session_name, NodeType=node_type).set(count)
        for node_type in node_types:
            count = autoscaler_summary.active_nodes.get(node_type, 0)
            self.prom_metrics.active_nodes.labels(SessionName=self.prom_metrics.session_name, NodeType=node_type).set(count)
        failed_node_counts = Counter()
        for _, node_type in autoscaler_summary.failed_nodes:
            failed_node_counts[node_type] += 1
        for node_type, count in failed_node_counts.items():
            self.prom_metrics.recently_failed_nodes.labels(SessionName=self.prom_metrics.session_name, NodeType=node_type).set(count)

    def update_event_summary(self):
        """Report the current size of the cluster.

        To avoid log spam, only cluster size changes (CPU, GPU or TPU count change)
        are reported to the event summarizer. The event summarizer will report
        only the latest cluster size per batch.
        """
        avail_resources = self.load_metrics.resources_avail_summary()
        if not self.readonly_config and avail_resources != self.last_avail_resources:
            self.event_summarizer.add('Resized to {}.', quantity=avail_resources, aggregate=lambda old, new: new)
            self.last_avail_resources = avail_resources

    def destroy_autoscaler_workers(self):
        """Cleanup the autoscaler, in case of an exception in the run() method.

        We kill the worker nodes, but retain the head node in order to keep
        logs around, keeping costs minimal. This monitor process runs on the
        head node anyway, so this is more reliable."""
        if self.autoscaler is None:
            return
        if self.autoscaling_config is None:
            logger.error('Monitor: Cleanup failed due to lack of autoscaler config.')
            return
        logger.info('Monitor: Exception caught. Taking down workers...')
        clean = False
        while not clean:
            try:
                teardown_cluster(config_file=self.autoscaling_config, yes=True, workers_only=True, override_cluster_name=None, keep_min_workers=True)
                clean = True
                logger.info('Monitor: Workers taken down.')
            except Exception:
                logger.error('Monitor: Cleanup exception. Trying again...')
                time.sleep(2)

    def _handle_failure(self, error):
        if self.autoscaler is not None and os.environ.get('RAY_AUTOSCALER_FATESHARE_WORKERS', '') == '1':
            self.autoscaler.kill_workers()
            self.destroy_autoscaler_workers()
        message = f'The autoscaler failed with the following error:\n{error}'
        if _internal_kv_initialized():
            _internal_kv_put(ray_constants.DEBUG_AUTOSCALING_ERROR, message, overwrite=True)
        gcs_publisher = ray._raylet.GcsPublisher(address=self.gcs_address)
        from ray._private.utils import publish_error_to_driver
        publish_error_to_driver(ray_constants.MONITOR_DIED_ERROR, message, gcs_publisher=gcs_publisher)

    def _signal_handler(self, sig, frame):
        try:
            self._handle_failure(f'Terminated with signal {sig}\n' + ''.join(traceback.format_stack(frame)))
        except Exception:
            logger.exception('Monitor: Failure in signal handler.')
        sys.exit(sig + 128)

    def run(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        try:
            if _internal_kv_initialized():
                _internal_kv_del(ray_constants.DEBUG_AUTOSCALING_ERROR)
            self._initialize_autoscaler()
            self._run()
        except Exception:
            logger.exception('Error in monitor loop')
            self._handle_failure(traceback.format_exc())
            raise