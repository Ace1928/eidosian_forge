import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def _append_by_get_put(self, relpath, bytes):
    full_data = StringIO()
    try:
        data = self.get(relpath)
        full_data.write(data.read())
    except transport.NoSuchFile:
        pass
    before = full_data.tell()
    full_data.write(bytes)
    full_data.seek(0)
    self.put_file(relpath, full_data)
    return before