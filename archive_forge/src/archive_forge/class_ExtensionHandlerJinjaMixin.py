from __future__ import annotations
from logging import Logger
from typing import TYPE_CHECKING, Any, cast
from jinja2.exceptions import TemplateNotFound
from jupyter_server.base.handlers import FileFindHandler
class ExtensionHandlerJinjaMixin:
    """Mixin class for ExtensionApp handlers that use jinja templating for
    template rendering.
    """

    def get_template(self, name: str) -> str:
        """Return the jinja template object for a given name"""
        try:
            env = f'{self.name}_jinja2_env'
            return cast(str, self.settings[env].get_template(name))
        except TemplateNotFound:
            return cast(str, super().get_template(name))