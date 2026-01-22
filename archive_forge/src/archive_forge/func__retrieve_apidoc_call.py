from __future__ import absolute_import, division, print_function
import errno
import glob
import json
import os
import re
import re
import re
def _retrieve_apidoc_call(self, path, safe=False):
    try:
        return self.http_call('get', path)
    except Exception:
        if not safe:
            raise
        return None