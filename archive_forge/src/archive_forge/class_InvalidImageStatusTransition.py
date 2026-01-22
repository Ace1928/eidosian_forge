import urllib.parse as urlparse
from glance.i18n import _
class InvalidImageStatusTransition(Invalid):
    message = _('Image status transition from %(cur_status)s to %(new_status)s is not allowed')