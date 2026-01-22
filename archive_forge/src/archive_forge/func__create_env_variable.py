from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.common.text.converters import to_text, to_native
from subprocess import Popen, PIPE
def _create_env_variable(argument_name):

    def f(value, arguments, env):
        env[argument_name] = value
    return f