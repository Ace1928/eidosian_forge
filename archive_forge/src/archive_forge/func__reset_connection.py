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
def _reset_connection(task_action: str, connection: ConnectionBase, host_context: t.Dict[str, t.Any], ignore_errors: bool=False) -> None:
    """Resets the connection handling any errors"""

    def _wrap_conn_err(func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except (AnsibleError, RequestException) as e:
            if ignore_errors:
                return False
            raise AnsibleError(e)
        return True
    if host_context['do_close_on_reset']:
        display.vvvv(f'{task_action}: closing connection plugin')
        try:
            success = _wrap_conn_err(connection.close)
        except Exception:
            host_context['do_close_on_reset'] = False
            raise
        host_context['do_close_on_reset'] = success
    display.vvvv(f'{task_action}: resetting connection plugin')
    try:
        _wrap_conn_err(connection.reset)
    except AttributeError:
        pass