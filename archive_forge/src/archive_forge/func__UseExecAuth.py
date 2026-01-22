from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import os
import subprocess
import sys
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
def _UseExecAuth():
    """Returns a bool noting if ExecAuth should be enabled."""
    env_flag = 'USE_GKE_GCLOUD_AUTH_PLUGIN'
    use_gke_gcloud_auth_plugin = encoding.GetEncodedValue(os.environ, env_flag)
    if use_gke_gcloud_auth_plugin is not None:
        if use_gke_gcloud_auth_plugin.lower() == 'false':
            return False
        elif use_gke_gcloud_auth_plugin.lower() != 'true':
            log.warning('Ignoring unsupported env value found for {}={}'.format(env_flag, use_gke_gcloud_auth_plugin.lower()))
    return True