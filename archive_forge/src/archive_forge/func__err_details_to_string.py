import datetime
import email.message
import math
from operator import methodcaller
import sys
import unittest
import warnings
from testtools.compat import _b
from testtools.content import (
from testtools.content_type import ContentType
from testtools.tags import TagContext
def _err_details_to_string(self, test, err=None, details=None):
    """Convert an error in exc_info form or a contents dict to a string."""
    if err is not None:
        return TracebackContent(err, test, capture_locals=self.tb_locals).as_text()
    return _details_to_str(details, special='traceback')