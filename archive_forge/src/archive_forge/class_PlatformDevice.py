import enum
import os
import sys
import requests
from six.moves.urllib import request
from tensorflow.python.eager import context
from tensorflow.python.platform import tf_logging as logging
@enum.unique
class PlatformDevice(enum.Enum):
    INTERNAL_CPU = 'internal_CPU'
    INTERNAL_GPU = 'internal_GPU'
    INTERNAL_TPU = 'internal_TPU'
    GCE_GPU = 'GCE_GPU'
    GCE_TPU = 'GCE_TPU'
    GCE_CPU = 'GCE_CPU'
    UNSUPPORTED = 'unsupported'