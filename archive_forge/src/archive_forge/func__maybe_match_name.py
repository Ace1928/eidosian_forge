from __future__ import annotations
from functools import wraps
from typing import (
from pandas._libs.lib import item_from_zerodim
from pandas._libs.missing import is_matching_na
from pandas.core.dtypes.generic import (
def _maybe_match_name(a, b):
    """
    Try to find a name to attach to the result of an operation between
    a and b.  If only one of these has a `name` attribute, return that
    name.  Otherwise return a consensus name if they match or None if
    they have different names.

    Parameters
    ----------
    a : object
    b : object

    Returns
    -------
    name : str or None

    See Also
    --------
    pandas.core.common.consensus_name_attr
    """
    a_has = hasattr(a, 'name')
    b_has = hasattr(b, 'name')
    if a_has and b_has:
        try:
            if a.name == b.name:
                return a.name
            elif is_matching_na(a.name, b.name):
                return a.name
            else:
                return None
        except TypeError:
            if is_matching_na(a.name, b.name):
                return a.name
            return None
        except ValueError:
            return None
    elif a_has:
        return a.name
    elif b_has:
        return b.name
    return None