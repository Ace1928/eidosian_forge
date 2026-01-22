import os
import tempfile
import testtools
from testtools.matchers import StartsWith
from fixtures import (
class ContrivedException(Exception):
    pass