import logging
import os
import shutil
import stat
import tarfile
import zipfile
from typing import Iterable, List, Optional
from zipfile import ZipInfo
from pip._internal.exceptions import InstallationError
from pip._internal.utils.filetypes import (
from pip._internal.utils.misc import ensure_dir
def unpack_file(filename: str, location: str, content_type: Optional[str]=None) -> None:
    filename = os.path.realpath(filename)
    if content_type == 'application/zip' or filename.lower().endswith(ZIP_EXTENSIONS) or zipfile.is_zipfile(filename):
        unzip_file(filename, location, flatten=not filename.endswith('.whl'))
    elif content_type == 'application/x-gzip' or tarfile.is_tarfile(filename) or filename.lower().endswith(TAR_EXTENSIONS + BZ2_EXTENSIONS + XZ_EXTENSIONS):
        untar_file(filename, location)
    else:
        logger.critical('Cannot unpack file %s (downloaded from %s, content-type: %s); cannot detect archive format', filename, location, content_type)
        raise InstallationError(f'Cannot determine archive format of {location}')