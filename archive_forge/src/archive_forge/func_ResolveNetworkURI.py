from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import six
def ResolveNetworkURI(project, network, resource_parser):
    """Resolves the URI of a network."""
    if project and network and resource_parser:
        return six.text_type(resource_parser.Parse(network, collection='compute.networks', params={'project': project}))
    return None