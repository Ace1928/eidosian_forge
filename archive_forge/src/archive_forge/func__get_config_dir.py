from __future__ import annotations
import logging
import os
import sys
import typing as t
from jupyter_core.application import JupyterApp
from jupyter_core.paths import ENV_CONFIG_PATH, SYSTEM_CONFIG_PATH, jupyter_config_dir
from tornado.log import LogFormatter
from traitlets import Bool
from jupyter_server._version import __version__
from jupyter_server.extension.config import ExtensionConfigManager
from jupyter_server.extension.manager import ExtensionManager, ExtensionPackage
def _get_config_dir(user: bool=False, sys_prefix: bool=False) -> str:
    """Get the location of config files for the current context

    Returns the string to the environment

    Parameters
    ----------
    user : bool [default: False]
        Get the user's .jupyter config directory
    sys_prefix : bool [default: False]
        Get sys.prefix, i.e. ~/.envs/my-env/etc/jupyter
    """
    if user and sys_prefix:
        sys_prefix = False
    if user:
        extdir = jupyter_config_dir()
    elif sys_prefix:
        extdir = ENV_CONFIG_PATH[0]
    else:
        extdir = SYSTEM_CONFIG_PATH[0]
    return extdir