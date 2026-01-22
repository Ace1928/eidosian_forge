from oslo_utils import reflection
import webob.exc
from heat.common.i18n import _
from heat.common import serializers
class HeatRequestLimitExceeded(HeatAPIException):
    """Payload size of the request exceeds maximum allowed size."""
    code = 400
    title = 'RequestLimitExceeded'
    explanation = _('Payload exceeds maximum allowed size')