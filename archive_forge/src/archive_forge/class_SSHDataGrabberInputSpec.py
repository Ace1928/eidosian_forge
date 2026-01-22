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
class SSHDataGrabberInputSpec(DataGrabberInputSpec):
    hostname = Str(mandatory=True, desc='Server hostname.')
    username = Str(desc='Server username.')
    password = traits.Password(desc='Server password.')
    download_files = traits.Bool(True, usedefault=True, desc='If false it will return the file names without downloading them')
    base_directory = Str(mandatory=True, desc='Path to the base directory consisting of subject data.')
    template_expression = traits.Enum(['fnmatch', 'regexp'], usedefault=True, desc='Use either fnmatch or regexp to express templates')
    ssh_log_to_file = Str('', usedefault=True, desc='If set SSH commands will be logged to the given file')