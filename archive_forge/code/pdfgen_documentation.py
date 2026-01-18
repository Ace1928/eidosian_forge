import os
import sys
import tempfile
import time
from io import StringIO
from math import ceil, cos, pi, sin, tan
from types import *
from . import pdfdoc, pdfgeom, pdfmetrics, pdfutils
prints multi-line or newlined strings, moving down.  One
        common use is to quote a multi-line block in your Python code;
        since this may be indented, by default it trims whitespace
        off each line and from the beginning; set trim=0 to preserve
        whitespace.