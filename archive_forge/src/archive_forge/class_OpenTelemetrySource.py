from __future__ import (absolute_import, division, print_function)
import getpass
import os
import socket
import sys
import time
import uuid
from collections import OrderedDict
from os.path import basename
from ansible.errors import AnsibleError
from ansible.module_utils.six import raise_from
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.plugins.callback import CallbackBase
class OpenTelemetrySource(object):

    def __init__(self, display):
        self.ansible_playbook = ''
        self.ansible_version = None
        self.session = str(uuid.uuid4())
        self.host = socket.gethostname()
        try:
            self.ip_address = socket.gethostbyname(socket.gethostname())
        except Exception as e:
            self.ip_address = None
        self.user = getpass.getuser()
        self._display = display

    def traceparent_context(self, traceparent):
        carrier = dict()
        carrier['traceparent'] = traceparent
        return TraceContextTextMapPropagator().extract(carrier=carrier)

    def start_task(self, tasks_data, hide_task_arguments, play_name, task):
        """ record the start of a task for one or more hosts """
        uuid = task._uuid
        if uuid in tasks_data:
            return
        name = task.get_name().strip()
        path = task.get_path()
        action = task.action
        args = None
        if not task.no_log and (not hide_task_arguments):
            args = task.args
        tasks_data[uuid] = TaskData(uuid, name, path, play_name, action, args)

    def finish_task(self, tasks_data, status, result, dump):
        """ record the results of a task for a single host """
        task_uuid = result._task._uuid
        if hasattr(result, '_host') and result._host is not None:
            host_uuid = result._host._uuid
            host_name = result._host.name
        else:
            host_uuid = 'include'
            host_name = 'include'
        task = tasks_data[task_uuid]
        if self.ansible_version is None and hasattr(result, '_task_fields') and result._task_fields['args'].get('_ansible_version'):
            self.ansible_version = result._task_fields['args'].get('_ansible_version')
        task.dump = dump
        task.add_host(HostData(host_uuid, host_name, status, result))

    def generate_distributed_traces(self, otel_service_name, ansible_playbook, tasks_data, status, traceparent, disable_logs, disable_attributes_in_logs):
        """ generate distributed traces from the collected TaskData and HostData """
        tasks = []
        parent_start_time = None
        for task_uuid, task in tasks_data.items():
            if parent_start_time is None:
                parent_start_time = task.start
            tasks.append(task)
        trace.set_tracer_provider(TracerProvider(resource=Resource.create({SERVICE_NAME: otel_service_name})))
        processor = BatchSpanProcessor(OTLPSpanExporter())
        trace.get_tracer_provider().add_span_processor(processor)
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(ansible_playbook, context=self.traceparent_context(traceparent), start_time=parent_start_time, kind=SpanKind.SERVER) as parent:
            parent.set_status(status)
            if self.ansible_version is not None:
                parent.set_attribute('ansible.version', self.ansible_version)
            parent.set_attribute('ansible.session', self.session)
            parent.set_attribute('ansible.host.name', self.host)
            if self.ip_address is not None:
                parent.set_attribute('ansible.host.ip', self.ip_address)
            parent.set_attribute('ansible.host.user', self.user)
            for task in tasks:
                for host_uuid, host_data in task.host_data.items():
                    with tracer.start_as_current_span(task.name, start_time=task.start, end_on_exit=False) as span:
                        self.update_span_data(task, host_data, span, disable_logs, disable_attributes_in_logs)

    def update_span_data(self, task_data, host_data, span, disable_logs, disable_attributes_in_logs):
        """ update the span with the given TaskData and HostData """
        name = '[%s] %s: %s' % (host_data.name, task_data.play, task_data.name)
        message = 'success'
        res = {}
        rc = 0
        status = Status(status_code=StatusCode.OK)
        if host_data.status != 'included':
            if 'results' in host_data.result._result:
                if host_data.status == 'failed':
                    message = self.get_error_message_from_results(host_data.result._result['results'], task_data.action)
                    enriched_error_message = self.enrich_error_message_from_results(host_data.result._result['results'], task_data.action)
            else:
                res = host_data.result._result
                rc = res.get('rc', 0)
                if host_data.status == 'failed':
                    message = self.get_error_message(res)
                    enriched_error_message = self.enrich_error_message(res)
            if host_data.status == 'failed':
                status = Status(status_code=StatusCode.ERROR, description=message)
                span.record_exception(BaseException(enriched_error_message))
            elif host_data.status == 'skipped':
                message = res['skip_reason'] if 'skip_reason' in res else 'skipped'
                status = Status(status_code=StatusCode.UNSET)
            elif host_data.status == 'ignored':
                status = Status(status_code=StatusCode.UNSET)
        span.set_status(status)
        attributes = {'ansible.task.module': task_data.action, 'ansible.task.message': message, 'ansible.task.name': name, 'ansible.task.result': rc, 'ansible.task.host.name': host_data.name, 'ansible.task.host.status': host_data.status}
        if isinstance(task_data.args, dict) and 'gather_facts' not in task_data.action:
            names = tuple((self.transform_ansible_unicode_to_str(k) for k in task_data.args.keys()))
            values = tuple((self.transform_ansible_unicode_to_str(k) for k in task_data.args.values()))
            attributes['ansible.task.args.name'] = names
            attributes['ansible.task.args.value'] = values
        self.set_span_attributes(span, attributes)
        self.add_attributes_for_service_map_if_possible(span, task_data)
        if not disable_logs:
            span.add_event(task_data.dump, attributes={} if disable_attributes_in_logs else attributes)
            span.end(end_time=host_data.finish)

    def set_span_attributes(self, span, attributes):
        """ update the span attributes with the given attributes if not None """
        if span is None and self._display is not None:
            self._display.warning('span object is None. Please double check if that is expected.')
        elif attributes is not None:
            span.set_attributes(attributes)

    def add_attributes_for_service_map_if_possible(self, span, task_data):
        """Update the span attributes with the service that the task interacted with, if possible."""
        redacted_url = self.parse_and_redact_url_if_possible(task_data.args)
        if redacted_url:
            span.set_attribute('http.url', redacted_url.geturl())

    @staticmethod
    def parse_and_redact_url_if_possible(args):
        """Parse and redact the url, if possible."""
        try:
            parsed_url = urlparse(OpenTelemetrySource.url_from_args(args))
        except ValueError:
            return None
        if OpenTelemetrySource.is_valid_url(parsed_url):
            return OpenTelemetrySource.redact_user_password(parsed_url)
        return None

    @staticmethod
    def url_from_args(args):
        url_args = ('url', 'api_url', 'baseurl', 'repo', 'server_url', 'chart_repo_url', 'registry_url', 'endpoint', 'uri', 'updates_url')
        for arg in url_args:
            if args is not None and args.get(arg):
                return args.get(arg)
        return ''

    @staticmethod
    def redact_user_password(url):
        return url._replace(netloc=url.hostname) if url.password else url

    @staticmethod
    def is_valid_url(url):
        if all([url.scheme, url.netloc, url.hostname]):
            return '{{' not in url.hostname
        return False

    @staticmethod
    def transform_ansible_unicode_to_str(value):
        parsed_url = urlparse(str(value))
        if OpenTelemetrySource.is_valid_url(parsed_url):
            return OpenTelemetrySource.redact_user_password(parsed_url).geturl()
        return str(value)

    @staticmethod
    def get_error_message(result):
        if result.get('exception') is not None:
            return OpenTelemetrySource._last_line(result['exception'])
        return result.get('msg', 'failed')

    @staticmethod
    def get_error_message_from_results(results, action):
        for result in results:
            if result.get('failed', False):
                return '{0}({1}) - {2}'.format(action, result.get('item', 'none'), OpenTelemetrySource.get_error_message(result))

    @staticmethod
    def _last_line(text):
        lines = text.strip().split('\n')
        return lines[-1]

    @staticmethod
    def enrich_error_message(result):
        message = result.get('msg', 'failed')
        exception = result.get('exception')
        stderr = result.get('stderr')
        return 'message: "{0}"\nexception: "{1}"\nstderr: "{2}"'.format(message, exception, stderr)

    @staticmethod
    def enrich_error_message_from_results(results, action):
        message = ''
        for result in results:
            if result.get('failed', False):
                message = '{0}({1}) - {2}\n{3}'.format(action, result.get('item', 'none'), OpenTelemetrySource.enrich_error_message(result), message)
        return message