from __future__ import annotations
import getpass
import os
import platform
import socket
import sys
from collections.abc import Callable
from functools import wraps
from importlib import reload
from typing import Any, Dict, Optional
from twisted.conch.ssh import keys
from twisted.python import failure, filepath, log, usage
def _inputSaveFile(prompt: str) -> str:
    """
    Ask the user where to save the key.

    This needs to be a separate function so the unit test can patch it.
    """
    return input(prompt)