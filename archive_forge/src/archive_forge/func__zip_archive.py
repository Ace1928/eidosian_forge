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
def _zip_archive(self, export_name, files, root):
    archname = '.'.join(self._unique_name(export_name, 'zip', root))
    with zipfile.ZipFile(os.path.join(root, archname), 'w') as zipf:
        for (basename, ext), entry in files:
            filename = self._truncate_name(basename, ext)
            zipf.writestr(f'{export_name}/{filename}', Exporter.encode(entry))