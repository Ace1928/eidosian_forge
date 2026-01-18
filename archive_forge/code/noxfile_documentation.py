import os
import pathlib
import subprocess
import shutil
import tempfile
from nox.command import which
import nox
Installs and configures the Cloud SDK with the given application default
    credentials.

    If project is True, then a project will be set in the active config.
    If it is false, this will ensure no project is set.
    