import urllib.parse as urlparse
from glance.i18n import _
class MetadefDuplicateResourceTypeAssociation(Duplicate):
    message = _('The metadata definition resource-type association of resource-type=%(resource_type_name)s to namespace=%(namespace_name)s already exists.')