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
def _getKeyOrDefault(options: Dict[Any, Any], inputCollector: Optional[Callable[[str], str]]=None, keyTypeName: str='rsa') -> str:
    """
    If C{options["filename"]} is None, prompt the user to enter a path
    or attempt to set it to .ssh/id_rsa
    @param options: command line options
    @param inputCollector: dependency injection for testing
    @param keyTypeName: key type or "rsa"
    """
    if inputCollector is None:
        inputCollector = input
    filename = options['filename']
    if not filename:
        filename = os.path.expanduser(f'~/.ssh/id_{keyTypeName}')
        if platform.system() == 'Windows':
            filename = os.path.expanduser(f'%HOMEPATH %\\.ssh\\id_{keyTypeName}')
        filename = inputCollector('Enter file in which the key is (%s): ' % filename) or filename
    return str(filename)