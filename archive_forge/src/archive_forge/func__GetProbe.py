from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import json
import textwrap
from typing import Mapping
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
def _GetProbe(probe, probe_type=''):
    """Returns the information message for the given probe."""
    if not probe:
        return ''
    probe_action = 'TCP'
    port = ''
    path = ''
    if probe.httpGet:
        probe_action = 'HTTP'
        path = probe.httpGet.path
    if probe.tcpSocket:
        probe_action = 'TCP'
        port = probe.tcpSocket.port
    if probe.grpc:
        probe_action = 'GRPC'
        port = probe.grpc.port
    return cp.Lines(['{probe_action} every {period}s'.format(probe_action=probe_action, period=probe.periodSeconds), cp.Labeled([('Path', path), ('Port', port), ('Initial delay', '{initial_delay}s'.format(initial_delay=probe.initialDelaySeconds or '0')), ('Timeout', '{timeout}s'.format(timeout=probe.timeoutSeconds)), ('Failure threshold', '{failures}'.format(failures=probe.failureThreshold)), ('Type', probe_type)])])