import urllib.parse as urlparse
from glance.i18n import _
class AuthBadRequest(GlanceException):
    message = _('Connect error/bad request to Auth service at URL %(url)s.')