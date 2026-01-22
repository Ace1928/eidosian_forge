import errno
import glob
import os
from pathlib import Path
from traitlets import Unicode, observe
from nbconvert.utils.io import link_or_copy
from .base import WriterBase
@observe('build_directory')
def _build_directory_changed(self, change):
    new = change['new']
    if new:
        self._makedir(new)