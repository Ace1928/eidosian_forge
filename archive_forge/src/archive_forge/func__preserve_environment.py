import os
import sys
import subprocess
import locale
import warnings
from numpy.distutils.misc_util import is_sequence, make_temp_file
from numpy.distutils import log
def _preserve_environment(names):
    log.debug('_preserve_environment(%r)' % names)
    env = {name: os.environ.get(name) for name in names}
    return env