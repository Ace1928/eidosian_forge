from oslo_utils import reflection
import webob.exc
from heat.common.i18n import _
from heat.common import serializers
class HeatThrottlingError(HeatAPIException):
    """Request was denied due to request throttling."""
    code = 400
    title = 'Throttling'
    explanation = _('Request was denied due to request throttling')