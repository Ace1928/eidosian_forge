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
def _do_until_success_or_timeout(task_action: str, connection: ConnectionBase, host_context: t.Dict[str, t.Any], action_desc: str, timeout: float, func: t.Callable[..., T], *args: t.Any, **kwargs: t.Any) -> t.Optional[T]:
    """Runs the function multiple times ignoring errors until a timeout occurs"""
    max_end_time = datetime.datetime.utcnow() + datetime.timedelta(seconds=timeout)

    def wait_condition(idx):
        return datetime.datetime.utcnow() < max_end_time
    try:
        return _do_until_success_or_condition(task_action, connection, host_context, action_desc, wait_condition, func, *args, **kwargs)
    except Exception:
        raise Exception('Timed out waiting for %s (timeout=%s)' % (action_desc, timeout))