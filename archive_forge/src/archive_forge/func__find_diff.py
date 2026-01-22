from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import translation
def _find_diff(self, update_prps, stored_prps):
    add_prps = list(set(update_prps or []) - set(stored_prps or []))
    remove_prps = list(set(stored_prps or []) - set(update_prps or []))
    return (add_prps, remove_prps)