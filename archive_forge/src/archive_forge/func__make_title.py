from __future__ import absolute_import
import inspect
from inspect import cleandoc, getdoc, getfile, isclass, ismodule, signature
from typing import Any, Collection, Iterable, Optional, Tuple, Type, Union
from .console import Group, RenderableType
from .control import escape_control_codes
from .highlighter import ReprHighlighter
from .jupyter import JupyterMixin
from .panel import Panel
from .pretty import Pretty
from .table import Table
from .text import Text, TextType
def _make_title(self, obj: Any) -> Text:
    """Make a default title."""
    title_str = str(obj) if isclass(obj) or callable(obj) or ismodule(obj) else str(type(obj))
    title_text = self.highlighter(title_str)
    return title_text