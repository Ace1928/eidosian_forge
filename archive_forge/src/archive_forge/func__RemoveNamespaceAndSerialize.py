from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.kuberun import structuredout
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kuberun import kuberun_command
def _RemoveNamespaceAndSerialize(data):
    return json.dumps({k: v for k, v in data.items() if k != 'namespace'}, sort_keys=True)