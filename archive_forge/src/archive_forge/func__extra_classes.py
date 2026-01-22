from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Sequence
from markdown_it import MarkdownIt
from markdown_it.rules_block import StateBlock
from mdit_py_plugins.utils import is_code_block
def _extra_classes(markup: str) -> list[str]:
    """Return the list of additional classes based on the markup."""
    if markup.startswith('?'):
        if markup.endswith('+'):
            return ['is-collapsible collapsible-open']
        return ['is-collapsible collapsible-closed']
    return []