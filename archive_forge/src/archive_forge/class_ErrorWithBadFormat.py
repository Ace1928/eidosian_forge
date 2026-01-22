import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
class ErrorWithBadFormat(errors.BzrError):
    _fmt = 'One format specifier: %(thing)s'