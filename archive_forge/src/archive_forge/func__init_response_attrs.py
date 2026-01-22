import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def _init_response_attrs(self):
    self.href = None
    self.length = -1
    self.executable = None
    self.is_dir = False