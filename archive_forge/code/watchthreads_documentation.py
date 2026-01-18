import sys
import cgi
import time
import traceback
from io import StringIO
from thread import get_ident
from paste import httpexceptions
from paste.request import construct_url, parse_formvars
from paste.util.template import HTMLTemplate, bunch

    Returns a plain-text traceback of the given thread, or None if it
    can't get a traceback.
    