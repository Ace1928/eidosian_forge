import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def annot_preprocess(page: 'Page') -> int:
    """Prepare for annotation insertion on the page.

    Returns:
        Old page rotation value. Temporarily sets rotation to 0 when required.
    """
    CheckParent(page)
    if not page.parent.is_pdf:
        raise ValueError('is no PDF')
    old_rotation = page.rotation
    if old_rotation != 0:
        page.set_rotation(0)
    return old_rotation