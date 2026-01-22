from __future__ import annotations
import collections.abc as c
import codecs
import ctypes.util
import fcntl
import getpass
import io
import logging
import os
import random
import subprocess
import sys
import termios
import textwrap
import threading
import time
import tty
import typing as t
from functools import wraps
from struct import unpack, pack
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleAssertionError, AnsiblePromptInterrupt, AnsiblePromptNoninteractive
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.six import text_type
from ansible.utils.color import stringc
from ansible.utils.multiprocessing import context as multiprocessing_context
from ansible.utils.singleton import Singleton
from ansible.utils.unsafe_proxy import wrap_var
def _read_non_blocking_stdin(self, echo: bool=False, seconds: int | None=None, interrupt_input: c.Container[bytes] | None=None, complete_input: c.Container[bytes] | None=None) -> bytes:
    if self._final_q:
        raise NotImplementedError
    if seconds is not None:
        start = time.time()
    if interrupt_input is None:
        try:
            interrupt = termios.tcgetattr(sys.stdin.buffer.fileno())[6][termios.VINTR]
        except Exception:
            interrupt = b'\x03'
    try:
        backspace_sequences = [termios.tcgetattr(self._stdin_fd)[6][termios.VERASE]]
    except Exception:
        backspace_sequences = [b'\x7f', b'\x08']
    result_string = b''
    while seconds is None or time.time() - start < seconds:
        key_pressed = None
        try:
            os.set_blocking(self._stdin_fd, False)
            while key_pressed is None and (seconds is None or time.time() - start < seconds):
                key_pressed = self._stdin.read(1)
                time.sleep(C.DEFAULT_INTERNAL_POLL_INTERVAL)
        finally:
            os.set_blocking(self._stdin_fd, True)
            if key_pressed is None:
                key_pressed = b''
        if interrupt_input is None and key_pressed == interrupt or (interrupt_input is not None and key_pressed.lower() in interrupt_input):
            clear_line(self._stdout)
            raise AnsiblePromptInterrupt('user interrupt')
        if complete_input is None and key_pressed in (b'\r', b'\n') or (complete_input is not None and key_pressed.lower() in complete_input):
            clear_line(self._stdout)
            break
        elif key_pressed in backspace_sequences:
            clear_line(self._stdout)
            result_string = result_string[:-1]
            if echo:
                self._stdout.write(result_string)
            self._stdout.flush()
        else:
            result_string += key_pressed
    return result_string