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
@bothmethod
def _merge_metadata(self_or_cls, obj, fn, *dicts):
    """
        Returns a merged metadata info dictionary from the supplied
        function and additional dictionaries
        """
    merged = {k: v for d in dicts for k, v in d.items()}
    return dict(merged, **fn(obj)) if fn else merged