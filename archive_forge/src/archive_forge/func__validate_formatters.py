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
def _validate_formatters(self):
    if not self.parse_fields(self.filename_formatter).issubset(self.ffields):
        raise Exception(f'Valid filename fields are: {','.join(sorted(self.ffields))}')
    elif not self.parse_fields(self.export_name).issubset(self.efields):
        raise Exception(f'Valid export fields are: {','.join(sorted(self.efields))}')
    try:
        time.strftime(self.timestamp_format, tuple(time.localtime()))
    except Exception as e:
        raise Exception('Timestamp format invalid') from e