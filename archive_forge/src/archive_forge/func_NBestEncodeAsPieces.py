from sys import version_info as _swig_python_version_info
import re
import csv
import sys
import os
from io import StringIO
from io import BytesIO
from ._version import __version__
def NBestEncodeAsPieces(self, input, nbest_size=None, **kwargs):
    return self.NBestEncode(input=input, nbest_size=nbest_size, out_type=str, **kwargs)