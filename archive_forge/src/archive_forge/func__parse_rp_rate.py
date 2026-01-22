import uuid
import os_traits
from neutron_lib._i18n import _
from neutron_lib import constants as const
from neutron_lib.placement import constants as place_const
def _parse_rp_rate(rate_str):
    """Parse the config string value to an non-negative integer.

    :param rate_str: A decimal string representation of an non-negative
                     integer, allowing the empty string.
    :raises: ValueError on invalid input.
    :returns: The value as an integer or None if not set in config.
    """
    try:
        rate = None
        if rate_str:
            rate = int(rate_str)
            if rate < 0:
                raise ValueError()
    except ValueError as e:
        raise ValueError(_('Cannot parse string value to an integer. Expected: non-negative integer value, got: %s') % rate_str) from e
    return rate