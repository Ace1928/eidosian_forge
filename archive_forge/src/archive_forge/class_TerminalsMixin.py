from __future__ import annotations
from typing import TYPE_CHECKING
from jupyter_server.extension.handler import ExtensionHandlerMixin
class TerminalsMixin(ExtensionHandlerMixin):
    """An extension mixin for terminals."""

    @property
    def terminal_manager(self) -> TerminalManager:
        return self.settings['terminal_manager']