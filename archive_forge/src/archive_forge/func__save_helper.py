from __future__ import annotations
import logging # isort:skip
from os.path import abspath, expanduser
from typing import Sequence
from jinja2 import Template
from ..core.templates import FILE
from ..core.types import PathLike
from ..models.ui import UIElement
from ..resources import Resources, ResourcesLike
from ..settings import settings
from ..themes import Theme
from ..util.warnings import warn
from .state import State, curstate
from .util import default_filename
def _save_helper(obj: UIElement | Sequence[UIElement], filename: PathLike, resources: Resources | None, title: str | None, template: Template | str | None, theme: Theme | None=None) -> None:
    """

    """
    from ..embed import file_html
    html = file_html(obj, resources=resources, title=title, template=template or FILE, theme=theme)
    with open(filename, mode='w', encoding='utf-8') as f:
        f.write(html)