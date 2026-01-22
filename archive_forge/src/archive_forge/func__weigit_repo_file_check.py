import copy
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union
import torch
from torch import Tensor
from .pygit import PyGit
from .sha1_store import SHA1_Store
def _weigit_repo_file_check(self, check_dir: Path) -> tuple:
    """Returns a tuple of boolean corresponding to the existence of each
        .wgit internally required files.
        """
    wgit_exists = check_dir.joinpath('.wgit').exists()
    git_exists = check_dir.joinpath('.wgit/.git').exists()
    gitignore_exists = check_dir.joinpath('.wgit/.gitignore').exists()
    return (wgit_exists, git_exists, gitignore_exists)