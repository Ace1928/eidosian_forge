import logging
import os
import select
import socket
import subprocess
import sys
from contextlib import closing
from io import BufferedReader, BytesIO
from typing import (
from urllib.parse import quote as urlquote
from urllib.parse import unquote as urlunquote
from urllib.parse import urljoin, urlparse, urlunparse, urlunsplit
import dulwich
from .config import Config, apply_instead_of, get_xdg_config_home_path
from .errors import GitProtocolError, NotGitRepository, SendPackError
from .pack import (
from .protocol import (
from .refs import PEELED_TAG_SUFFIX, _import_remote_refs, read_info_refs
from .repo import Repo
class PLinkSSHVendor(SSHVendor):
    """SSH vendor that shells out to the local 'plink' command."""

    def run_command(self, host, command, username=None, port=None, password=None, key_filename=None, ssh_command=None):
        if ssh_command:
            import shlex
            args = [*shlex.split(ssh_command, posix=sys.platform != 'win32'), '-ssh']
        elif sys.platform == 'win32':
            args = ['plink.exe', '-ssh']
        else:
            args = ['plink', '-ssh']
        if password is not None:
            import warnings
            warnings.warn('Invoking PLink with a password exposes the password in the process list.')
            args.extend(['-pw', str(password)])
        if port:
            args.extend(['-P', str(port)])
        if key_filename:
            args.extend(['-i', str(key_filename)])
        if username:
            host = f'{username}@{host}'
        if host.startswith('-'):
            raise StrangeHostname(hostname=host)
        args.append(host)
        proc = subprocess.Popen([*args, command], bufsize=0, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return SubprocessWrapper(proc)