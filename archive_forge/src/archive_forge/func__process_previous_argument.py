import abc
import argparse
import functools
import logging
from cliff import command
from cliff import lister
from cliff import show
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
def _process_previous_argument(current_arg, _value_number, current_type_str, _list_flag, _values_specs, _clear_flag, values_specs):
    if current_arg is not None:
        if _value_number == 0 and (current_type_str or _list_flag):
            raise exceptions.CommandError(_('Invalid values_specs %s') % ' '.join(values_specs))
        if _value_number > 1 or _list_flag or current_type_str == 'list':
            current_arg.update({'nargs': '+'})
        elif _value_number == 0:
            if _clear_flag:
                _values_specs.pop()
            else:
                current_arg.update({'action': 'store_true'})