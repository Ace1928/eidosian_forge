import urllib
from castellan.i18n import _
class KeyManagerError(CastellanException):
    message = _('Key manager error: %(reason)s')