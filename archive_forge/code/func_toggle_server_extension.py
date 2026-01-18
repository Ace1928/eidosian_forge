from __future__ import annotations
import logging
import os
import sys
import typing as t
from jupyter_core.application import JupyterApp
from jupyter_core.paths import ENV_CONFIG_PATH, SYSTEM_CONFIG_PATH, jupyter_config_dir
from tornado.log import LogFormatter
from traitlets import Bool
from jupyter_server._version import __version__
from jupyter_server.extension.config import ExtensionConfigManager
from jupyter_server.extension.manager import ExtensionManager, ExtensionPackage
def toggle_server_extension(self, import_name: str) -> None:
    """Change the status of a named server extension.

        Uses the value of `self._toggle_value`.

        Parameters
        ---------

        import_name : str
            Importable Python module (dotted-notation) exposing the magic-named
            `load_jupyter_server_extension` function
        """
    config_dir, extension_manager = _get_extmanager_for_context(user=self.user, sys_prefix=self.sys_prefix)
    try:
        self.log.info(f'{self._toggle_pre_message.capitalize()}: {import_name}')
        self.log.info(f'- Writing config: {config_dir}')
        self.log.info(f'    - Validating {import_name}...')
        extpkg = ExtensionPackage(name=import_name)
        extpkg.validate()
        version = extpkg.version
        self.log.info(f'      {import_name} {version} {GREEN_OK}')
        config = extension_manager.config_manager
        if config:
            if self._toggle_value is True:
                config.enable(import_name)
            else:
                config.disable(import_name)
        self.log.info(f'    - Extension successfully {self._toggle_post_message}.')
    except Exception as err:
        self.log.info(f'     {RED_X} Validation failed: {err}')