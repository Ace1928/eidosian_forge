import functools
from enum import Enum
from urllib.parse import urlencode
def to_routing_header(params, qualified_enums=True):
    """Returns a routing header string for the given request parameters.

    Args:
        params (Mapping[str, str | bytes | Enum]): A dictionary containing the request
            parameters used for routing.
        qualified_enums (bool): Whether to represent enum values
            as their type-qualified symbol names instead of as their
            unqualified symbol names.

    Returns:
        str: The routing header string.
    """
    tuples = params.items() if isinstance(params, dict) else params
    if not qualified_enums:
        tuples = [(x[0], x[1].name) if isinstance(x[1], Enum) else x for x in tuples]
    return '&'.join([_urlencode_param(*t) for t in tuples])