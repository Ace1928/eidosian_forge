from __future__ import absolute_import
import argparse
import os
import io
import json
import logging
import sys
import errno
import hashlib
import math
import shutil
import tempfile
from functools import partial
def _create_base_dir():
    """Create the gensim-data directory in home directory, if it has not been already created.

    Raises
    ------
    Exception
        An exception is raised when read/write permissions are not available or a file named gensim-data
        already exists in the home directory.

    """
    if not os.path.isdir(BASE_DIR):
        try:
            logger.info('Creating %s', BASE_DIR)
            os.makedirs(BASE_DIR)
        except OSError as e:
            if e.errno == errno.EEXIST:
                raise Exception('Not able to create folder gensim-data in {}. File gensim-data exists in the directory already.'.format(_PARENT_DIR))
            else:
                raise Exception("Can't create {}. Make sure you have the read/write permissions to the directory or you can try creating the folder manually".format(BASE_DIR))