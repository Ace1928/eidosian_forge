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
class BIDSDataGrabberInputSpec(DynamicTraitedSpec):
    base_dir = Directory(exists=True, desc='Path to BIDS Directory.', mandatory=True)
    output_query = traits.Dict(key_trait=Str, value_trait=traits.Dict, desc='Queries for outfield outputs')
    load_layout = Directory(exists=True, desc='Path to load already saved Bidslayout.', mandatory=False)
    raise_on_empty = traits.Bool(True, usedefault=True, desc='Generate exception if list is empty for a given field')
    index_derivatives = traits.Bool(False, mandatory=True, usedefault=True, desc='Index derivatives/ sub-directory')
    extra_derivatives = traits.List(Directory(exists=True), desc='Additional derivative directories to index')