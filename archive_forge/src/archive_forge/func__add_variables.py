from __future__ import absolute_import, division, print_function
import os
import os.path
import re
import shutil
import subprocess
import tempfile
import time
import shlex
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.parsing.convert_bool import BOOLEANS_FALSE
from ansible.module_utils.common.text.converters import to_text, to_bytes
@staticmethod
def _add_variables(variables_dict, build_command):
    """Return a command list with all found options.

        :param variables_dict: Pre-parsed optional variables used from a
                               seed command.
        :type variables_dict: ``dict``
        :param build_command: Command to run.
        :type build_command: ``list``
        :returns: list of command options.
        :rtype: ``list``
        """
    for key, value in variables_dict.items():
        build_command.append(str(key))
        build_command.append(str(value))
    return build_command