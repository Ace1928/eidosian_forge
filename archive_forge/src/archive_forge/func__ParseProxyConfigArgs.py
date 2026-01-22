import copy
import json
import os
from googlecloudsdk.command_lib.anthos.common import messages
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import exceptions as c_except
from googlecloudsdk.core.credentials import store as c_store
def _ParseProxyConfigArgs(self, proxy_config_type, pod_name_namespace, context, **kwargs):
    del kwargs
    exec_args = ['proxy-config', proxy_config_type, pod_name_namespace, '--context', context]
    return exec_args