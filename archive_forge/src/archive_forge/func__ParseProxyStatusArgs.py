import copy
import json
import os
from googlecloudsdk.command_lib.anthos.common import messages
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import exceptions as c_except
from googlecloudsdk.core.credentials import store as c_store
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