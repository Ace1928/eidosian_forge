import glob
from itertools import chain
import os
import sys
from traitlets.config.application import boolean_flag
from traitlets.config.configurable import Configurable
from traitlets.config.loader import Config
from IPython.core.application import SYSTEM_CONFIG_DIRS, ENV_CONFIG_DIRS
from IPython.core import pylabtools
from IPython.utils.contexts import preserve_keys
from IPython.utils.path import filefind
from traitlets import (
from IPython.terminal import pt_inputhooks
Run module specified at the command-line.