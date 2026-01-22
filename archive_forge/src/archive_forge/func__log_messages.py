from __future__ import (annotations, absolute_import, division, print_function)
import collections.abc as c
import fcntl
import io
import os
import shlex
import typing as t
from abc import abstractmethod
from functools import wraps
from ansible import constants as C
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.playbook.play_context import PlayContext
from ansible.plugins import AnsiblePlugin
from ansible.plugins.become import BecomeBase
from ansible.plugins.shell import ShellBase
from ansible.utils.display import Display
from ansible.plugins.loader import connection_loader, get_shell_plugin
from ansible.utils.path import unfrackpath
def _log_messages(self, message: str) -> None:
    if self.get_option('persistent_log_messages'):
        self.queue_message('log', message)