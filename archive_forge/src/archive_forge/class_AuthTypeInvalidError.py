import urllib
from castellan.i18n import _
class AuthTypeInvalidError(CastellanException):
    message = _('Invalid auth_type was specified, auth_type: %(type)s')