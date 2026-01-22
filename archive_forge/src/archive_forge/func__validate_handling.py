import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def _validate_handling(self):
    if self.dir_content is not None:
        self.expected_content_handled = True