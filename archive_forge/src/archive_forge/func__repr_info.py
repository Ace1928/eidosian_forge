import contextvars
import os
import socket
import subprocess
import sys
import threading
from . import format_helpers
def _repr_info(self):
    info = super()._repr_info()
    pos = 2 if self._cancelled else 1
    info.insert(pos, f'when={self._when}')
    return info