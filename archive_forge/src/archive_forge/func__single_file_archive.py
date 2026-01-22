import itertools
import os
import pickle
import re
import shutil
import string
import tarfile
import time
import zipfile
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import param
from param.parameterized import bothmethod
from .dimension import LabelledData
from .element import Collator, Element
from .ndmapping import NdMapping, UniformNdMapping
from .options import Store
from .overlay import Layout, Overlay
from .util import group_sanitizer, label_sanitizer, unique_iterator
def _single_file_archive(self, export_name, files, root):
    (basename, ext), entry = files[0]
    full_fname = f'{export_name}_{basename}'
    unique_name, ext = self._unique_name(full_fname, ext, root)
    filename = self._truncate_name(self._normalize_name(unique_name), ext=ext)
    fpath = os.path.join(root, filename)
    with open(fpath, 'wb') as f:
        f.write(Exporter.encode(entry))