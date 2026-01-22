import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def append_file(self, relpath, f, mode=None):
    """See Transport.append_file"""
    return self.append_bytes(relpath, f.read(), mode=mode)