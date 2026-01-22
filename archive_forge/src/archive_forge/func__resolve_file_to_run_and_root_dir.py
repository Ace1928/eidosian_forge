from __future__ import annotations
import datetime
import errno
import gettext
import hashlib
import hmac
import ipaddress
import json
import logging
import mimetypes
import os
import pathlib
import random
import re
import select
import signal
import socket
import stat
import sys
import threading
import time
import typing as t
import urllib
import warnings
from base64 import encodebytes
from pathlib import Path
import jupyter_client
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client.manager import KernelManager
from jupyter_client.session import Session
from jupyter_core.application import JupyterApp, base_aliases, base_flags
from jupyter_core.paths import jupyter_runtime_dir
from jupyter_events.logger import EventLogger
from nbformat.sign import NotebookNotary
from tornado import httpserver, ioloop, web
from tornado.httputil import url_concat
from tornado.log import LogFormatter, access_log, app_log, gen_log
from tornado.netutil import bind_sockets
from tornado.routing import Matcher, Rule
from traitlets import (
from traitlets.config import Config
from traitlets.config.application import boolean_flag, catch_config_error
from jupyter_server import (
from jupyter_server._sysinfo import get_sys_info
from jupyter_server._tz import utcnow
from jupyter_server.auth.authorizer import AllowAllAuthorizer, Authorizer
from jupyter_server.auth.identity import (
from jupyter_server.auth.login import LoginHandler
from jupyter_server.auth.logout import LogoutHandler
from jupyter_server.base.handlers import (
from jupyter_server.extension.config import ExtensionConfigManager
from jupyter_server.extension.manager import ExtensionManager
from jupyter_server.extension.serverextension import ServerExtensionApp
from jupyter_server.gateway.connections import GatewayWebSocketConnection
from jupyter_server.gateway.gateway_client import GatewayClient
from jupyter_server.gateway.managers import (
from jupyter_server.log import log_request
from jupyter_server.services.config import ConfigManager
from jupyter_server.services.contents.filemanager import (
from jupyter_server.services.contents.largefilemanager import AsyncLargeFileManager
from jupyter_server.services.contents.manager import AsyncContentsManager, ContentsManager
from jupyter_server.services.kernels.connection.base import BaseKernelWebsocketConnection
from jupyter_server.services.kernels.connection.channels import ZMQChannelsWebsocketConnection
from jupyter_server.services.kernels.kernelmanager import (
from jupyter_server.services.sessions.sessionmanager import SessionManager
from jupyter_server.utils import (
from jinja2 import Environment, FileSystemLoader
from jupyter_core.paths import secure_write
from jupyter_core.utils import ensure_async
from jupyter_server.transutils import _i18n, trans
from jupyter_server.utils import pathname2url, urljoin
def _resolve_file_to_run_and_root_dir(self) -> str:
    """Returns a relative path from file_to_run
        to root_dir. If root_dir and file_to_run
        are incompatible, i.e. on different subtrees,
        crash the app and log a critical message. Note
        that if root_dir is not configured and file_to_run
        is configured, root_dir will be set to the parent
        directory of file_to_run.
        """
    rootdir_abspath = pathlib.Path(self.root_dir).absolute()
    file_rawpath = pathlib.Path(self.file_to_run)
    combined_path = (rootdir_abspath / file_rawpath).absolute()
    is_child = str(combined_path).startswith(str(rootdir_abspath))
    if is_child:
        if combined_path.parent != rootdir_abspath:
            self.log.debug("The `root_dir` trait is set to a directory that's not the immediate parent directory of `file_to_run`. Note that the server will start at `root_dir` and open the the file from the relative path to the `root_dir`.")
        return str(combined_path.relative_to(rootdir_abspath))
    self.log.critical("`root_dir` and `file_to_run` are incompatible. They don't share the same subtrees. Make sure `file_to_run` is on the same path as `root_dir`.")
    self.exit(1)
    return ''