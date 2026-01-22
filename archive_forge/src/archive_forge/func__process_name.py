import glob
import fnmatch
import string
import json
import os
import os.path as op
import shutil
import subprocess
import re
import copy
import tempfile
from os.path import join, dirname
from warnings import warn
from .. import config, logging
from ..utils.filemanip import (
from ..utils.misc import human_order_sorted, str2bool
from .base import (
def _process_name(self, name, val):
    if '.' in name:
        newkeys = name.split('.')
        name = newkeys.pop(0)
        nested_dict = {newkeys.pop(): val}
        for nk in reversed(newkeys):
            nested_dict = {nk: nested_dict}
        val = nested_dict
    return (name, val)