import urllib.parse as urlparse
from glance.i18n import _
class MetadefResourceTypeAssociationNotFound(NotFound):
    message = _('The metadata definition resource-type association of resource-type=%(resource_type_name)s to namespace=%(namespace_name)s, was not found.')