from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import shutil
import tempfile
import unittest
from gae_ext_runtime import ext_runtime
class InvalidRuntime(Exception):
    """Raised when the runtime directory is doesn't match the runtime."""