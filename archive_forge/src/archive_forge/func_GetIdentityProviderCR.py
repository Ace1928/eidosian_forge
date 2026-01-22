from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import json
import re
from googlecloudsdk.api_lib.container import util as c_util
from googlecloudsdk.command_lib.compute.instance_templates import service_proxy_aux_data
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import kube_util as hub_kube_util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def GetIdentityProviderCR(self):
    """Get the YAML output of the IdentityProvider CR."""
    if not self._IdentityProviderCRDExists():
        raise ClusterError('IdentityProvider CRD is not found in the cluster. Please install Anthos Service Mesh with VM support and retry.')
    out, err = self._RunKubectl(['get', 'identityproviders.security.cloud.google.com', 'google', '-o', 'yaml'], None)
    if err:
        if 'NotFound' in err:
            raise ClusterError('GCE identity provider is not found in the cluster. Please install Anthos Service Mesh with VM support.')
        raise exceptions.Error('Error retrieving IdentityProvider google in default namespace: {}'.format(err))
    return out