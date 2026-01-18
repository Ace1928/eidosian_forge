from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import logging
import os
import re
import textwrap
import sys
import gslib
from gslib.utils.system_util import IS_OSX
from gslib.exception import CommandException
from gslib.storage_url import StorageUrlFromString
from gslib.utils.constants import GSUTIL_PUB_TARBALL
from gslib.utils.constants import GSUTIL_PUB_TARBALL_PY2
Returns the appropriate gsutil pub tarball based on the Python version.

  Returns:
    The storage_uri of the appropriate pub tarball.
  