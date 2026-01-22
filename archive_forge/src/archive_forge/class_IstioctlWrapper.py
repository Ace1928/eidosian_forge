import copy
import json
import os
from googlecloudsdk.command_lib.anthos.common import messages
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import exceptions as c_except
from googlecloudsdk.core.credentials import store as c_store
class IstioctlWrapper(binary_operations.StreamingBinaryBackedOperation):
    """`istioctl_backend` wrapper."""

    def __init__(self, **kwargs):
        custom_errors = {'MISSING_EXEC': messages.MISSING_BINARY.format(binary='istioctl')}
        super(IstioctlWrapper, self).__init__(binary='istioctl', custom_errors=custom_errors, **kwargs)

    def _ParseArgsForCommand(self, command, **kwargs):
        if command == 'bug-report':
            return self._ParseBugReportArgs(**kwargs)
        elif command == 'proxy-config':
            return self._ParseProxyConfigArgs(**kwargs)
        elif command == 'proxy-status':
            return self._ParseProxyStatusArgs(**kwargs)

    def _ParseBugReportArgs(self, context, **kwargs):
        del kwargs
        exec_args = ['bug-report', '--context', context]
        return exec_args

    def _ParseProxyConfigArgs(self, proxy_config_type, pod_name_namespace, context, **kwargs):
        del kwargs
        exec_args = ['proxy-config', proxy_config_type, pod_name_namespace, '--context', context]
        return exec_args

    def _ParseProxyStatusArgs(self, context, pod_name, mesh_name, project_number, **kwargs):
        del kwargs
        exec_args = ['experimental', 'proxy-status', '--xds-via-agents']
        if pod_name:
            exec_args.extend([pod_name])
        exec_args.extend(['--context', context])
        if mesh_name:
            exec_args.extend(['--meshName', 'mesh:' + mesh_name])
        if project_number:
            exec_args.extend(['--projectNumber', project_number])
        return exec_args