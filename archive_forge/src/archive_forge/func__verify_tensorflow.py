from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import json
import sys
def _verify_tensorflow(version):
    """Check whether TensorFlow is installed at an appropriate version."""
    try:
        import tensorflow.compat.v1 as tf
    except ImportError:
        eprint('Cannot import Tensorflow. Please verify "python -c \'import tensorflow\'" works.')
        return False
    try:
        if tf.__version__ < version:
            eprint('Tensorflow version must be at least {} .'.format(version), VERIFY_TENSORFLOW_VERSION)
            return False
    except (NameError, AttributeError) as e:
        eprint('Error while getting the installed TensorFlow version: ', e, '\n', VERIFY_TENSORFLOW_VERSION)
        return False
    return True