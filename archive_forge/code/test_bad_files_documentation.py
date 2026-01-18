import os
from io import StringIO
from .. import errors
from ..status import show_tree_status
from . import TestCaseWithTransport
from .features import OsFifoFeature
Test that bzr will ignore files it doesn't like