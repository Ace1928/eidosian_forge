import uuid
import os_traits
from neutron_lib._i18n import _
from neutron_lib import constants as const
from neutron_lib.placement import constants as place_const
def _parse_rp_options(options, dict_keys):
    """Parse the config string tuples and map them to dict of dicts.

    :param options: The list of string tuples separated with ':'
                    in following format '[<str>]:[<rate0>:...:<rateN>]'.
                    First element of the tuple is a string that will be used as
                    a key of an outer dictionary. If not specified, defaults to
                    an empty string. A rate is an optional, non-negative int.
                    If rate values are provided they must match the length of
                    dict_keys. If a rate value is not specified, it defaults to
                    None.
    :param dict_keys: A tuple of strings containing names of inner dictionary
                      keys that are going to be mapped to rate values from
                      options tuple.
    :raises: ValueError on invalid input.
    :returns: The fully parsed config as a dict of dicts.
    """
    rv = {}
    for option in options:
        if ':' not in option:
            option += ':' * len(dict_keys)
        try:
            values = option.split(':')
            tuple_len = len(dict_keys) + 1
            if len(values) != tuple_len:
                raise ValueError()
            name = values.pop(0)
        except ValueError as e:
            raise ValueError(_('Expected a tuple with %d values, got: %s') % (tuple_len, option)) from e
        if name in rv:
            raise ValueError(_('Same resource name listed multiple times: "%s"') % name)
        rv[name] = dict(zip(dict_keys, [_parse_rp_rate(v) for v in values]))
    return rv