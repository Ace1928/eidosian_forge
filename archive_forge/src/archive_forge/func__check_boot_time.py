import datetime
import json
import random
import re
import time
import traceback
import uuid
import typing as t
from ansible.errors import AnsibleConnectionFailure, AnsibleError
from ansible.module_utils.common.text.converters import to_text
from ansible.plugins.connection import ConnectionBase
from ansible.utils.display import Display
def _check_boot_time(task_action: str, connection: ConnectionBase, host_context: t.Dict[str, t.Any], previous_boot_time: int, boot_time_command: str, timeout: int):
    """Checks the system boot time has been changed or not"""
    display.vvvv('%s: attempting to get system boot time' % task_action)
    if timeout:
        _set_connection_timeout(task_action, connection, host_context, timeout)
    current_boot_time = _get_system_boot_time(task_action, connection, boot_time_command)
    if current_boot_time == previous_boot_time:
        raise _TestCommandFailure('boot time has not changed')