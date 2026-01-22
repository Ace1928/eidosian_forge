from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from collections import namedtuple
import errno
import json
import random
import six
import gslib
from gslib.exception import CommandException
from gslib.tracker_file import (WriteJsonDataToTrackerFile,
from gslib.utils.constants import UTF8
class _CompositeUploadTrackerEntry(object):
    """Enum class for composite upload tracker file JSON keys."""
    COMPONENTS_LIST = 'components'
    COMPONENT_NAME = 'component_name'
    COMPONENT_GENERATION = 'component_generation'
    ENC_SHA256 = 'encryption_key_sha256'
    PREFIX = 'prefix'