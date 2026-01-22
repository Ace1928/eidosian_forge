from __future__ import annotations
import json
import os
from glob import glob
from typing import Any
import json5
from jsonschema import Draft7Validator as Validator
from jsonschema import ValidationError
from jupyter_server import _tz as tz
from jupyter_server.base.handlers import APIHandler
from jupyter_server.services.config.manager import ConfigManager, recursive_update
from tornado import web
from .translation_utils import DEFAULT_LOCALE, L10N_SCHEMA_NAME, SYS_LOCALE, is_valid_locale
class SchemaHandler(APIHandler):
    """Base handler for handler requiring access to settings."""

    def initialize(self, app_settings_dir: str, schemas_dir: str, settings_dir: str, labextensions_path: list[str] | None, **kwargs: Any) -> None:
        """Initialize the handler."""
        super().initialize(**kwargs)
        self.overrides, error = _get_overrides(app_settings_dir)
        self.app_settings_dir = app_settings_dir
        self.schemas_dir = schemas_dir
        self.settings_dir = settings_dir
        self.labextensions_path = labextensions_path
        if error:
            overrides_warning = 'Failed loading overrides: %s'
            self.log.warning(overrides_warning, error)

    def get_current_locale(self) -> str:
        """
        Get the current locale as specified in the translation-extension settings.

        Returns
        -------
        str
            The current locale string.

        Notes
        -----
        If the locale setting is not available or not valid, it will default to jupyterlab_server.translation_utils.DEFAULT_LOCALE.
        """
        try:
            settings, _ = get_settings(self.app_settings_dir, self.schemas_dir, self.settings_dir, schema_name=L10N_SCHEMA_NAME, overrides=self.overrides, labextensions_path=self.labextensions_path)
        except web.HTTPError as e:
            schema_warning = 'Missing or misshapen translation settings schema:\n%s'
            self.log.warning(schema_warning, e)
            settings = {}
        current_locale = settings.get('settings', {}).get('locale') or SYS_LOCALE
        if current_locale == 'default':
            current_locale = SYS_LOCALE
        if not is_valid_locale(current_locale):
            current_locale = DEFAULT_LOCALE
        return current_locale