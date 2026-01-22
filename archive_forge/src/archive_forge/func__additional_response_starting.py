import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def _additional_response_starting(self, name):
    """A additional response element inside a multistatus begins."""
    pass