from keystoneauth1 import exceptions as ksa_exceptions
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.i18n import _
def _check_integer(self, value, msg=None):
    """Attempt to convert value to an integer

        Raises InvalidValue on failure

        :param value:
            Convert this to an integer.  None is converted to 0 (zero).
        :param msg:
            An alternate message for the exception, must include exactly
            one substitution to receive the attempted value.
        """
    if value is None:
        return 0
    try:
        value = int(value)
    except (TypeError, ValueError):
        if not msg:
            msg = _('%s is not an integer') % value
        raise InvalidValue(msg)
    return value