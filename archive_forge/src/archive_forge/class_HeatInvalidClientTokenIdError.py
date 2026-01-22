from oslo_utils import reflection
import webob.exc
from heat.common.i18n import _
from heat.common import serializers
class HeatInvalidClientTokenIdError(HeatAPIException):
    """The X.509 certificate or AWS Access Key ID provided does not exist."""
    code = 403
    title = 'InvalidClientTokenId'
    explanation = _('The certificate or AWS Key ID provided does not exist')