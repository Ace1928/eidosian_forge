import collections
from heat.common.i18n import _
from heat.common import exception
from heat.engine import function
def _check_condition_type(self, condition_name, condition_defn):
    if not isinstance(condition_defn, (bool, function.Function)):
        msg_data = {'cd': condition_name, 'definition': condition_defn}
        message = _('The definition of condition "%(cd)s" is invalid: %(definition)s') % msg_data
        raise exception.StackValidationFailed(error='Condition validation error', message=message)