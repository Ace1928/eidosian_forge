from oslo_utils import reflection
import webob.exc
from heat.common.i18n import _
from heat.common import serializers
class HeatInvalidQueryParameterError(HeatAPIException):
    """AWS query string is malformed, does not adhere to AWS standards."""
    code = 400
    title = 'InvalidQueryParameter'
    explanation = _('AWS query string is malformed, does not adhere to AWS spec')