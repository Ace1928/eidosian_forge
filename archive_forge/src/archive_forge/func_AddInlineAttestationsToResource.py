from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from containerregistry.client import docker_name
from googlecloudsdk.core.exceptions import Error
import six
from six.moves import urllib
def AddInlineAttestationsToResource(resource, attestations):
    """Inlines attestations into a Kubernetes resource.

  Args:
    resource: The Kubernetes resource provided by the user.
    attestations: List of attestations returned by the policy evaluator in comma
      separated DSSE form.

  Returns:
    Modified Kubernetes resource with attestations inlined.
  """
    if resource['kind'] != 'Pod':
        resource['spec']['template'] = AddInlineAttestationsToPodSpec(resource['spec']['template'], attestations)
        return resource
    return AddInlineAttestationsToPodSpec(resource, attestations)