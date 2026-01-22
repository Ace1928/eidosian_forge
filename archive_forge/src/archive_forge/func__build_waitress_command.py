import importlib
import importlib.metadata
import os
import shlex
import sys
import textwrap
import types
from flask import Flask, Response, send_from_directory
from packaging.version import Version
from mlflow.exceptions import MlflowException
from mlflow.server import handlers
from mlflow.server.handlers import (
from mlflow.utils.os import get_entry_points, is_windows
from mlflow.utils.process import _exec_cmd
from mlflow.version import VERSION
def _build_waitress_command(waitress_opts, host, port, app_name, is_factory):
    opts = shlex.split(waitress_opts) if waitress_opts else []
    return [sys.executable, '-m', 'waitress', *opts, f'--host={host}', f'--port={port}', '--ident=mlflow', *(['--call'] if is_factory else []), app_name]