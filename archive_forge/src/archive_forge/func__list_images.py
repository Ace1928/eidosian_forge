import os
import gzip
import tarfile
import struct
import warnings
import numpy as np
from .. import dataset
from ...utils import download, check_sha1, _get_repo_file_url
from .... import nd, image, recordio, base
from .... import numpy as _mx_np  # pylint: disable=reimported
from ....util import is_np_array
def _list_images(self, root):
    self.synsets = []
    self.items = []
    for folder in sorted(os.listdir(root)):
        path = os.path.join(root, folder)
        if not os.path.isdir(path):
            warnings.warn('Ignoring %s, which is not a directory.' % path, stacklevel=3)
            continue
        label = len(self.synsets)
        self.synsets.append(folder)
        for filename in sorted(os.listdir(path)):
            filename = os.path.join(path, filename)
            ext = os.path.splitext(filename)[1]
            if ext.lower() not in self._exts:
                warnings.warn('Ignoring %s of type %s. Only support %s' % (filename, ext, ', '.join(self._exts)))
                continue
            self.items.append((filename, label))