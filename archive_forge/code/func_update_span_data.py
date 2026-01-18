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