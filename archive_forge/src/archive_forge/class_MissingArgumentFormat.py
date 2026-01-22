from __future__ import absolute_import, division, print_function
import os
from functools import wraps
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six import iteritems
class MissingArgumentFormat(CmdRunnerException):

    def __init__(self, arg, args_order, args_formats):
        self.args_order = args_order
        self.arg = arg
        self.args_formats = args_formats

    def __repr__(self):
        return 'MissingArgumentFormat({0!r}, {1!r}, {2!r})'.format(self.arg, self.args_order, self.args_formats)

    def __str__(self):
        return 'Cannot find format for parameter {0} {1} in: {2}'.format(self.arg, self.args_order, self.args_formats)