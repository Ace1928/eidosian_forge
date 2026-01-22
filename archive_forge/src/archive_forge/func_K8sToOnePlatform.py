from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import properties
def K8sToOnePlatform(service_resource, region):
    """Convert the Kubernetes-style service resource to One Platform-style."""
    project = properties.VALUES.core.project.Get(required=True)
    parts = kubernetes_ref.match(service_resource.RelativeName())
    service = parts.group('SERVICE')
    return 'projects/{project}/locations/{location}/services/{service}'.format(project=project, location=region, service=service)