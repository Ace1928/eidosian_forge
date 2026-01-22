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
class XNATSourceInputSpec(DynamicTraitedSpec, BaseInterfaceInputSpec):
    query_template = Str(mandatory=True, desc='Layout used to get files. Relative to base directory if defined')
    query_template_args = traits.Dict(Str, traits.List(traits.List), value=dict(outfiles=[]), usedefault=True, desc='Information to plug into template')
    server = Str(mandatory=True, requires=['user', 'pwd'], xor=['config'])
    user = Str()
    pwd = traits.Password()
    config = File(mandatory=True, xor=['server'])
    cache_dir = Directory(desc='Cache directory')