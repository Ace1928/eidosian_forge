from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.api_lib.cloudbuild.v2 import client_util
from googlecloudsdk.api_lib.cloudbuild.v2 import input_util
from googlecloudsdk.core import yaml
def _WorkspaceBindingTransform(workspace_binding):
    """Transform workspace binding message."""
    if 'secretName' in workspace_binding:
        popped_secret = workspace_binding.pop('secretName')
        workspace_binding['secret'] = {}
        workspace_binding['secret']['secretName'] = popped_secret
    elif 'volume' in workspace_binding:
        popped_volume = workspace_binding.pop('volume')
        workspace_binding['volumeClaim'] = {}
        if 'storage' in popped_volume:
            storage = popped_volume.pop('storage')
            workspace_binding['volumeClaim']['storage'] = storage
    else:
        return