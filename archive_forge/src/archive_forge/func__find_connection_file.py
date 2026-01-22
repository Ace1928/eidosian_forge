from __future__ import annotations
import json
import sys
from subprocess import PIPE, Popen
from typing import TYPE_CHECKING, Any
import jupyter_client
from jupyter_client import write_connection_file
def _find_connection_file(connection_file):
    """Return the absolute path for a connection file

    - If nothing specified, return current Kernel's connection file
    - Otherwise, call jupyter_client.find_connection_file
    """
    if connection_file is None:
        return get_connection_file()
    return jupyter_client.find_connection_file(connection_file)