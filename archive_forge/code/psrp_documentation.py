from __future__ import (annotations, absolute_import, division, print_function)
import base64
import json
import logging
import os
import typing as t
from ansible import constants as C
from ansible.errors import AnsibleConnectionFailure, AnsibleError
from ansible.errors import AnsibleFileNotFound
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.plugins.connection import ConnectionBase
from ansible.plugins.shell.powershell import ShellModule as PowerShellPlugin
from ansible.plugins.shell.powershell import _common_args
from ansible.utils.display import Display
from ansible.utils.hashing import sha1

        PSRP doesn't have the same concept as other protocols with its output.
        We need some extra logic to convert the pipeline streams and host
        output into the format that Ansible understands.

        :param pipeline: The finished PowerShell pipeline that invoked our
            commands
        :return: rc, stdout, stderr based on the pipeline output
        