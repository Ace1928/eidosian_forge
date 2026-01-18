import os
import shutil
import errno
from pathlib import Path
from traitlets.config.configurable import LoggingConfigurable
from ..paths import get_ipython_package_dir
from ..utils.path import expand_path, ensure_dir_exists
from traitlets import Unicode, Bool, observe
Find/create a profile dir and return its ProfileDir.

        This will create the profile directory if it doesn't exist.

        Parameters
        ----------
        profile_dir : unicode or str
            The path of the profile directory.
        