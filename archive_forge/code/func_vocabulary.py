import os
import zipfile
import shutil
import numpy as np
from . import _constants as C
from ...data import dataset
from ...utils import download, check_sha1, _get_repo_file_url
from ....contrib import text
from .... import nd, base
@property
def vocabulary(self):
    return self._vocab