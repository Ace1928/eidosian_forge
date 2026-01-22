from oslo_utils import reflection
import webob.exc
from heat.common.i18n import _
from heat.common import serializers
class HeatMissingParameterError(HeatAPIException):
    """A mandatory input parameter is missing.

    An input parameter that is mandatory for processing the request is missing.
    """
    code = 400
    title = 'MissingParameter'
    explanation = _('A mandatory input parameter is missing')