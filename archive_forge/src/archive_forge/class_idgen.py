import pathlib
from os import path
from pprint import pformat
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from jinja2 import BaseLoader, FileSystemLoader, TemplateNotFound
from jinja2.environment import Environment
from jinja2.sandbox import SandboxedEnvironment
from jinja2.utils import open_if_exists
from sphinx.application import TemplateBridge
from sphinx.theming import Theme
from sphinx.util import logging
from sphinx.util.osutil import mtimes_of_files
class idgen:

    def __init__(self) -> None:
        self.id = 0

    def current(self) -> int:
        return self.id

    def __next__(self) -> int:
        self.id += 1
        return self.id
    next = __next__