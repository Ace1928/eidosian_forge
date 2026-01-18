import configparser
import getpass
import os
import tempfile
from typing import Any, Optional
from wandb import env
from wandb.old import core
from wandb.sdk.lib import filesystem
from wandb.sdk.lib.runid import generate_id
def write_setting(settings, settings_path, persist):
    if not settings.has_section(section):
        Settings._safe_add_section(settings, Settings.DEFAULT_SECTION)
    settings.set(section, key, str(value))
    if persist:
        self._persist_settings(settings, settings_path)