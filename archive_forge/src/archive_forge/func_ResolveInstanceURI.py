from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import six
def ResolveInstanceURI(project, instance, resource_parser):
    """Resolves the URI of an instance."""
    if project and instance and resource_parser:
        return six.text_type(resource_parser.Parse(instance, collection='compute.instances', params={'project': project}))
    return None