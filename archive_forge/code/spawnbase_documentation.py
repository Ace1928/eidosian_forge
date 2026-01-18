from io import StringIO, BytesIO
import codecs
import os
import sys
import re
import errno
from .exceptions import ExceptionPexpect, EOF, TIMEOUT
from .expect import Expecter, searcher_string, searcher_re
Overridden in subclass using tty