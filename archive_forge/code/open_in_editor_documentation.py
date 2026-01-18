from __future__ import annotations
from prompt_toolkit.filters import emacs_mode, has_selection, vi_navigation_mode
from ..key_bindings import KeyBindings, KeyBindingsBase, merge_key_bindings
from .named_commands import get_by_name

    Pressing 'v' in navigation mode will open the buffer in an external editor.
    