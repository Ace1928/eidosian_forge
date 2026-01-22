from __future__ import absolute_import, division, print_function
import argparse
import contextlib
import numpy as np
class OverrideDefaultListAction(argparse.Action):
    """
    OverrideDefaultListAction

    An argparse action that works similarly to the regular 'append' action.
    The default value is deleted when a new value is specified. The 'append'
    action would append the new value to the default.

    Parameters
    ----------
    sep : str, optional
        Separator to be used if multiple values should be parsed from a list.

    """

    def __init__(self, sep=None, *args, **kwargs):
        super(OverrideDefaultListAction, self).__init__(*args, **kwargs)
        self.set_to_default = True
        self.list_type = self.type
        if sep is not None:
            self.type = str
        self.sep = sep

    def __call__(self, parser, namespace, value, option_string=None):
        if self.set_to_default:
            setattr(namespace, self.dest, [])
            self.set_to_default = False
        cur_values = getattr(namespace, self.dest)
        try:
            cur_values.extend([self.list_type(v) for v in value.split(self.sep)])
        except ValueError as e:
            raise argparse.ArgumentError(self, str(e) + value)