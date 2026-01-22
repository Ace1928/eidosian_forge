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
def _get_resource_properties(self):
    """
        page list Resource/Properties
        """
    page = self._pdf_page()
    ASSERT_PDF(page)
    rc = JM_get_resource_properties(page.obj())
    return rc