from __future__ import annotations
import functools
import importlib
import json
import logging
import mimetypes
import os
import pathlib
import re
import textwrap
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import (
import param
from bokeh.embed.bundle import (
from bokeh.model import Model
from bokeh.models import ImportedStyleSheet
from bokeh.resources import Resources as BkResources, _get_server_urls
from bokeh.settings import settings as _settings
from jinja2.environment import Environment
from jinja2.loaders import FileSystemLoader
from markupsafe import Markup
from ..config import config, panel_extension as extension
from ..util import isurl, url_path
from .state import state
def _collect_external_resources(self, resource_attr: ResourceAttr) -> list[str]:
    """ Collect external resources set on resource_attr attribute of all models."""
    external_resources: list[str] = []
    if state._extensions is not None:
        external_modules = {module: ext for ext, module in extension._imports.items()}
    else:
        external_modules = None
    for _, cls in sorted(Model.model_class_reverse_map.items(), key=lambda arg: arg[0]):
        if external_modules is not None and cls.__module__ in external_modules:
            if external_modules[cls.__module__] not in state._extensions:
                continue
        external: list[str] | str | None = getattr(cls, resource_attr, None)
        if isinstance(external, str):
            if external not in external_resources:
                external_resources.append(external)
        elif isinstance(external, list):
            for e in external:
                if e not in external_resources:
                    external_resources.append(e)
    return external_resources