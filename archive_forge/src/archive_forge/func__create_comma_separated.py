from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.common.text.converters import to_text, to_native
from subprocess import Popen, PIPE
def _create_comma_separated(argument_name):

    def f(value, arguments, env):
        arguments.extend([argument_name, ','.join([to_native(v) for v in value])])
    return f