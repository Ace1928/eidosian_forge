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
class DataFinderInputSpec(DynamicTraitedSpec, BaseInterfaceInputSpec):
    root_paths = traits.Either(traits.List(), Str(), mandatory=True)
    match_regex = Str('(.+)', usedefault=True, desc='Regular expression for matching paths.')
    ignore_regexes = traits.List(desc='List of regular expressions, if any match the path it will be ignored.')
    max_depth = traits.Int(desc='The maximum depth to search beneath the root_paths')
    min_depth = traits.Int(desc='The minimum depth to search beneath the root paths')
    unpack_single = traits.Bool(False, usedefault=True, desc='Unpack single results from list')