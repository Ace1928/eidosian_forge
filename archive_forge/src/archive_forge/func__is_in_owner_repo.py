from __future__ import annotations
import functools
import logging
import os
import subprocess
import sys
import warnings
from datetime import datetime
from typing import TYPE_CHECKING
def _is_in_owner_repo() -> bool:
    """Check if is running in code owner's repo.
            Only generate reliable check when `git` is installed and remote name
            is "origin".
            """
    try:
        result = subprocess.run(['git', 'config', '--get', 'remote.origin.url'], stdout=subprocess.PIPE)
        owner_repo = result.stdout.decode('utf-8').strip().lstrip('https://github.com/').lstrip('git@github.com:').rstrip('.git')
        return owner_repo == os.getenv('GITHUB_REPOSITORY')
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False