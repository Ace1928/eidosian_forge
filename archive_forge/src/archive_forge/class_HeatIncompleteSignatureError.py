from oslo_utils import reflection
import webob.exc
from heat.common.i18n import _
from heat.common import serializers
class HeatIncompleteSignatureError(HeatAPIException):
    """The request signature does not conform to AWS standards."""
    code = 400
    title = 'IncompleteSignature'
    explanation = _('The request signature does not conform to AWS standards')