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
def _get_formatted_doc(self, object_: Any) -> Optional[str]:
    """
        Extract the docstring of an object, process it and returns it.
        The processing consists in cleaning up the doctring's indentation,
        taking only its 1st paragraph if `self.help` is not True,
        and escape its control codes.

        Args:
            object_ (Any): the object to get the docstring from.

        Returns:
            Optional[str]: the processed docstring, or None if no docstring was found.
        """
    docs = getdoc(object_)
    if docs is None:
        return None
    docs = cleandoc(docs).strip()
    if not self.help:
        docs = _first_paragraph(docs)
    return escape_control_codes(docs)