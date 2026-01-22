import copy
import functools
import os
import sys
from oslo_utils import encodeutils
from oslo_utils import strutils
from glanceclient.common import progressbar
from glanceclient.common import utils
from glanceclient import exc
import glanceclient.v1.images
def _is_image_data_provided(args):
    """Return True if some image data has probably been provided by the user"""
    try:
        os.fstat(0)
    except OSError:
        return False
    return not sys.stdin.isatty() or args.file or args.copy_from