from oslo_utils import reflection
import webob.exc
from heat.common.i18n import _
from heat.common import serializers
class HeatSignatureError(HeatAPIException):
    """Authentication fails due to a bad signature."""
    code = 403
    title = 'SignatureDoesNotMatch'
    explanation = _('The request signature we calculated does not match the signature you provided')