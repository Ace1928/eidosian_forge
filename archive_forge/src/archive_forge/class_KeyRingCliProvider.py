import logging
import os
import shutil
import subprocess
import sysconfig
import typing
import urllib.parse
from abc import ABC, abstractmethod
from functools import lru_cache
from os.path import commonprefix
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
from pip._vendor.requests.auth import AuthBase, HTTPBasicAuth
from pip._vendor.requests.models import Request, Response
from pip._vendor.requests.utils import get_netrc_auth
from pip._internal.utils.logging import getLogger
from pip._internal.utils.misc import (
from pip._internal.vcs.versioncontrol import AuthInfo
class KeyRingCliProvider(KeyRingBaseProvider):
    """Provider which uses `keyring` cli

    Instead of calling the keyring package installed alongside pip
    we call keyring on the command line which will enable pip to
    use which ever installation of keyring is available first in
    PATH.
    """
    has_keyring = True

    def __init__(self, cmd: str) -> None:
        self.keyring = cmd

    def get_auth_info(self, url: str, username: Optional[str]) -> Optional[AuthInfo]:
        if username is not None:
            password = self._get_password(url, username)
            if password is not None:
                return (username, password)
        return None

    def save_auth_info(self, url: str, username: str, password: str) -> None:
        return self._set_password(url, username, password)

    def _get_password(self, service_name: str, username: str) -> Optional[str]:
        """Mirror the implementation of keyring.get_password using cli"""
        if self.keyring is None:
            return None
        cmd = [self.keyring, 'get', service_name, username]
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        res = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, env=env)
        if res.returncode:
            return None
        return res.stdout.decode('utf-8').strip(os.linesep)

    def _set_password(self, service_name: str, username: str, password: str) -> None:
        """Mirror the implementation of keyring.set_password using cli"""
        if self.keyring is None:
            return None
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        subprocess.run([self.keyring, 'set', service_name, username], input=f'{password}{os.linesep}'.encode('utf-8'), env=env, check=True)
        return None