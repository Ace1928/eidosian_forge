from __future__ import annotations
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from traitlets.config.loader import JSONFileConfigLoader, PyFileConfigLoader
from traitlets.log import get_logger
from .application import JupyterApp
from .paths import jupyter_config_dir, jupyter_data_dir
from .utils import ensure_dir_exists
class JupyterMigrate(JupyterApp):
    """A Jupyter Migration App."""
    name = 'jupyter-migrate'
    description = '\n    Migrate configuration and data from .ipython prior to 4.0 to Jupyter locations.\n\n    This migrates:\n\n    - config files in the default profile\n    - kernels in ~/.ipython/kernels\n    - notebook javascript extensions in ~/.ipython/extensions\n    - custom.js/css to .jupyter/custom\n\n    to their new Jupyter locations.\n\n    All files are copied, not moved.\n    If the destinations already exist, nothing will be done.\n    '

    def start(self) -> None:
        """Start the application."""
        if not migrate():
            self.log.info('Found nothing to migrate.')