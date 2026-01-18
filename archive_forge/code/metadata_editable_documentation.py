import os
from pip._vendor.pyproject_hooks import BuildBackendHookCaller
from pip._internal.build_env import BuildEnvironment
from pip._internal.exceptions import (
from pip._internal.utils.subprocess import runner_with_spinner_message
from pip._internal.utils.temp_dir import TempDirectory
Generate metadata using mechanisms described in PEP 660.

    Returns the generated metadata directory.
    