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
def _todim(val: Union[int, str]) -> str:
    """
    Make val a css dimension. In particular the following transformations
    are performed:

    - None -> 'initial' (default CSS value)
    - 0 -> '0'
    - ints and string representations of ints are interpreted as pixels.

    Everything else is returned unchanged.
    """
    if val is None:
        return 'initial'
    elif str(val).isdigit():
        return '0' if int(val) == 0 else '%spx' % val
    return val