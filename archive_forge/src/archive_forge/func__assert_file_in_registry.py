import os
import time
import contextlib
from pathlib import Path
import shlex
import shutil
from .hashes import hash_matches, file_hash
from .utils import (
from .downloaders import DOIDownloader, choose_downloader, doi_to_repository
def _assert_file_in_registry(self, fname):
    """
        Check if a file is in the registry and raise :class:`ValueError` if
        it's not.
        """
    if fname not in self.registry:
        raise ValueError(f"File '{fname}' is not in the registry.")