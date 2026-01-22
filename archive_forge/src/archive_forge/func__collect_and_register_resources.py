import functools
import os
import sys
import collections
import importlib
import warnings
from contextvars import copy_context
from importlib.machinery import ModuleSpec
import pkgutil
import threading
import re
import logging
import time
import mimetypes
import hashlib
import base64
import traceback
from urllib.parse import urlparse
from typing import Dict, Optional, Union
import flask
from importlib_metadata import version as _get_distribution_version
from dash import dcc
from dash import html
from dash import dash_table
from .fingerprint import build_fingerprint, check_fingerprint
from .resources import Scripts, Css
from .dependencies import (
from .development.base_component import ComponentRegistry
from .exceptions import (
from .version import __version__
from ._configs import get_combined_config, pathname_configs, pages_folder_config
from ._utils import (
from . import _callback
from . import _get_paths
from . import _dash_renderer
from . import _validate
from . import _watch
from . import _get_app
from ._grouping import map_grouping, grouping_len, update_args_group
from . import _pages
from ._pages import (
from ._jupyter import jupyter_dash, JupyterDisplayMode
from .types import RendererHooks
def _collect_and_register_resources(self, resources, include_async=True):

    def _relative_url_path(relative_package_path='', namespace=''):
        if any((relative_package_path.startswith(x + '/') for x in ['dcc', 'html', 'dash_table'])):
            relative_package_path = relative_package_path.replace('dash.', '')
            version = importlib.import_module(f'{namespace}.{os.path.split(relative_package_path)[0]}').__version__
        else:
            version = importlib.import_module(namespace).__version__
        module_path = os.path.join(os.path.dirname(sys.modules[namespace].__file__), relative_package_path)
        modified = int(os.stat(module_path).st_mtime)
        fingerprint = build_fingerprint(relative_package_path, version, modified)
        return f'{self.config.requests_pathname_prefix}_dash-component-suites/{namespace}/{fingerprint}'
    srcs = []
    for resource in resources:
        is_dynamic_resource = resource.get('dynamic', False)
        is_async = resource.get('async') is not None
        excluded = not include_async and is_async
        if 'relative_package_path' in resource:
            paths = resource['relative_package_path']
            paths = [paths] if isinstance(paths, str) else paths
            for rel_path in paths:
                if any((x in rel_path for x in ['dcc', 'html', 'dash_table'])):
                    rel_path = rel_path.replace('dash.', '')
                self.registered_paths[resource['namespace']].add(rel_path)
                if not is_dynamic_resource and (not excluded):
                    srcs.append(_relative_url_path(relative_package_path=rel_path, namespace=resource['namespace']))
        elif 'external_url' in resource:
            if not is_dynamic_resource and (not excluded):
                if isinstance(resource['external_url'], str):
                    srcs.append(resource['external_url'])
                else:
                    srcs += resource['external_url']
        elif 'absolute_path' in resource:
            raise Exception("Serving files from absolute_path isn't supported yet")
        elif 'asset_path' in resource:
            static_url = self.get_asset_url(resource['asset_path'])
            if static_url.endswith('.mjs'):
                srcs.append({'src': static_url + f'?m={resource['ts']}', 'type': 'module'})
            else:
                srcs.append(static_url + f'?m={resource['ts']}')
    return srcs