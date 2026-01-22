import urllib.parse as urlparse
from glance.i18n import _
class MetadefDuplicateObject(Duplicate):
    message = _('A metadata definition object with name=%(object_name)s already exists in namespace=%(namespace_name)s.')