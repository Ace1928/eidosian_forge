from __future__ import absolute_import, division, print_function
import os
from functools import wraps
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six import iteritems
class MissingArgumentValue(CmdRunnerException):

    def __init__(self, args_order, arg):
        self.args_order = args_order
        self.arg = arg

    def __repr__(self):
        return 'MissingArgumentValue({0!r}, {1!r})'.format(self.args_order, self.arg)

    def __str__(self):
        return 'Cannot find value for parameter {0} in {1}'.format(self.arg, self.args_order)