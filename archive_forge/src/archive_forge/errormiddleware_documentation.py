import sys
import traceback
import cgi
from io import StringIO
from paste.exceptions import formatter, collector, reporter
from paste import wsgilib
from paste import request
Close and return any error message