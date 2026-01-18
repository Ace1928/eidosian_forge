from __future__ import annotations
import logging # isort:skip
from ..core.templates import SCRIPT_TAG
from ..util.strings import indent
def wrap_in_script_tag(js: str, type: str='text/javascript', id: str | None=None) -> str:
    """

    """
    return SCRIPT_TAG.render(js_code=indent(js, 2), type=type, id=id)