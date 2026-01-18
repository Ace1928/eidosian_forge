import unittest
from os import sys, path
def test_option_types(self, error_msg, option, val):
    for tp, available in (((str, list), ('implies', 'headers', 'flags', 'group', 'detect', 'extra_checks')), ((str,), ('disable',)), ((int,), ('interest',)), ((bool,), ('implies_detect',)), ((bool, type(None)), ('autovec',))):
        found_it = option in available
        if not found_it:
            continue
        if not isinstance(val, tp):
            error_tp = [t.__name__ for t in (*tp,)]
            error_tp = ' or '.join(error_tp)
            raise AssertionError(error_msg + "expected '%s' type for option '%s' not '%s'" % (error_tp, option, type(val).__name__))
        break
    if not found_it:
        raise AssertionError(error_msg + "invalid option name '%s'" % option)