import re
from prometheus_client import start_http_server
from prometheus_client.core import (
from opencensus.common.transports import sync
from opencensus.stats import aggregation_data as aggregation_data_module
from opencensus.stats import base_exporter
import logging
class Collector(object):
    """Collector represents the Prometheus Collector object"""

    def __init__(self, options=Options(), view_name_to_data_map=None):
        if view_name_to_data_map is None:
            view_name_to_data_map = {}
        self._options = options
        self._registry = options.registry
        self._view_name_to_data_map = view_name_to_data_map
        self._registered_views = {}

    @property
    def options(self):
        """Options to be used to configure the exporter"""
        return self._options

    @property
    def registry(self):
        """Prometheus Collector Registry instance"""
        return self._registry

    @property
    def view_name_to_data_map(self):
        """Map with all view data objects
        that will be sent to Prometheus
        """
        return self._view_name_to_data_map

    @property
    def registered_views(self):
        """Map with all registered views"""
        return self._registered_views

    def register_view(self, view):
        """register_view will create the needed structure
        in order to be able to sent all data to Prometheus
        """
        v_name = get_view_name(self.options.namespace, view)
        if v_name not in self.registered_views:
            desc = {'name': v_name, 'documentation': view.description, 'labels': list(map(sanitize, view.columns)), 'units': view.measure.unit}
            self.registered_views[v_name] = desc

    def add_view_data(self, view_data):
        """Add view data object to be sent to server"""
        self.register_view(view_data.view)
        v_name = get_view_name(self.options.namespace, view_data.view)
        self.view_name_to_data_map[v_name] = view_data

    def to_metric(self, desc, tag_values, agg_data, metrics_map):
        """to_metric translate the data that OpenCensus create
        to Prometheus format, using Prometheus Metric object
        :type desc: dict
        :param desc: The map that describes view definition
        :type tag_values: tuple of :class:
            `~opencensus.tags.tag_value.TagValue`
        :param object of opencensus.tags.tag_value.TagValue:
            TagValue object used as label values
        :type agg_data: object of :class:
            `~opencensus.stats.aggregation_data.AggregationData`
        :param object of opencensus.stats.aggregation_data.AggregationData:
            Aggregated data that needs to be converted as Prometheus samples
        :rtype: :class:`~prometheus_client.core.CounterMetricFamily` or
                :class:`~prometheus_client.core.HistogramMetricFamily` or
                :class:`~prometheus_client.core.UnknownMetricFamily` or
                :class:`~prometheus_client.core.GaugeMetricFamily`
        :returns: A Prometheus metric object
        """
        metric_name = desc['name']
        metric_description = desc['documentation']
        label_keys = desc['labels']
        metric_units = desc['units']
        assert len(tag_values) == len(label_keys), (tag_values, label_keys)
        tag_values = [tv if tv else '' for tv in tag_values]
        if isinstance(agg_data, aggregation_data_module.CountAggregationData):
            metric = metrics_map.get(metric_name)
            if not metric:
                metric = CounterMetricFamily(name=metric_name, documentation=metric_description, unit=metric_units, labels=label_keys)
                metrics_map[metric_name] = metric
            metric.add_metric(labels=tag_values, value=agg_data.count_data)
            return metric
        elif isinstance(agg_data, aggregation_data_module.DistributionAggregationData):
            assert agg_data.bounds == sorted(agg_data.bounds)
            buckets = []
            cum_count = 0
            for ii, bound in enumerate(agg_data.bounds):
                cum_count += agg_data.counts_per_bucket[ii]
                bucket = [str(bound), cum_count]
                buckets.append(bucket)
            buckets.append(['+Inf', agg_data.count_data])
            metric = metrics_map.get(metric_name)
            if not metric:
                metric = HistogramMetricFamily(name=metric_name, documentation=metric_description, labels=label_keys)
                metrics_map[metric_name] = metric
            metric.add_metric(labels=tag_values, buckets=buckets, sum_value=agg_data.sum)
            return metric
        elif isinstance(agg_data, aggregation_data_module.SumAggregationData):
            metric = metrics_map.get(metric_name)
            if not metric:
                metric = UnknownMetricFamily(name=metric_name, documentation=metric_description, labels=label_keys)
                metrics_map[metric_name] = metric
            metric.add_metric(labels=tag_values, value=agg_data.sum_data)
            return metric
        elif isinstance(agg_data, aggregation_data_module.LastValueAggregationData):
            metric = metrics_map.get(metric_name)
            if not metric:
                metric = GaugeMetricFamily(name=metric_name, documentation=metric_description, labels=label_keys)
                metrics_map[metric_name] = metric
            metric.add_metric(labels=tag_values, value=agg_data.value)
            return metric
        else:
            raise ValueError(f'unsupported aggregation type {type(agg_data)}')

    def collect(self):
        """Collect fetches the statistics from OpenCensus
        and delivers them as Prometheus Metrics.
        Collect is invoked every time a prometheus.Gatherer is run
        for example when the HTTP endpoint is invoked by Prometheus.
        """
        metrics_map = {}
        for v_name, view_data in self._view_name_to_data_map.copy().items():
            if v_name not in self.registered_views:
                continue
            desc = self.registered_views[v_name]
            for tag_values in view_data.tag_value_aggregation_data_map:
                agg_data = view_data.tag_value_aggregation_data_map[tag_values]
                metric = self.to_metric(desc, tag_values, agg_data, metrics_map)
        for metric in metrics_map.values():
            yield metric