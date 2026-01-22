import inspect
import sys
from magnumclient.i18n import _
class PaymentRequired(HTTPClientError):
    """HTTP 402 - Payment Required.

    Reserved for future use.
    """
    http_status = 402
    message = _('Payment Required')