import gettext
import io
import sys
import json
import logging
import threading
import tempfile
import os
import shutil
import signal
import socket
import webbrowser
import errno
import random
import jinja2
import tornado.ioloop
import tornado.web
from traitlets.config.application import Application
from traitlets.config.loader import Config
from traitlets import Unicode, Integer, Bool, Dict, List, Callable, default, Type, Bytes
from jupyter_server.services.contents.largefilemanager import LargeFileManager
from jupyter_server.services.kernels.handlers import KernelHandler
from jupyter_server.base.handlers import path_regex, FileFindHandler
from jupyter_server.config_manager import recursive_update
from jupyterlab_server.themes_handler import ThemesHandler
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_core.paths import jupyter_config_path, jupyter_path
from .paths import ROOT, STATIC_ROOT, collect_template_paths, collect_static_paths
from .handler import VoilaHandler
from .treehandler import VoilaTreeHandler
from ._version import __version__
from .static_file_handler import MultiStaticFileHandler, TemplateStaticFileHandler, WhiteListFileHandler
from .configuration import VoilaConfiguration
from .execute import VoilaExecutor
from .exporter import VoilaExporter
from .shutdown_kernel_handler import VoilaShutdownKernelHandler
from .voila_kernel_manager import voila_kernel_manager_factory
from .request_info_handler import RequestInfoSocketHandler
from .utils import create_include_assets_functions
def _init_asyncio_patch(self):
    """set default asyncio policy to be compatible with tornado
        Tornado 6 (at least) is not compatible with the default
        asyncio implementation on Windows
        Pick the older SelectorEventLoopPolicy on Windows
        if the known-incompatible default policy is in use.
        do this as early as possible to make it a low priority and overridable
        ref: https://github.com/tornadoweb/tornado/issues/2608
        FIXME: if/when tornado supports the defaults in asyncio,
               remove and bump tornado requirement for py38
        """
    if sys.platform.startswith('win') and sys.version_info >= (3, 8):
        import asyncio
        try:
            from asyncio import WindowsProactorEventLoopPolicy, WindowsSelectorEventLoopPolicy
        except ImportError:
            pass
        else:
            if type(asyncio.get_event_loop_policy()) is WindowsProactorEventLoopPolicy:
                asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())