import urllib.parse as urlparse
from glance.i18n import _
class MetadefPropertyNotFound(NotFound):
    message = _('The metadata definition property with name=%(property_name)s was not found in namespace=%(namespace_name)s.')