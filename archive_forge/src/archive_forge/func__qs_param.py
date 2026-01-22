from __future__ import absolute_import, division, print_function
import errno
import glob
import json
import os
import re
import re
import re
def _qs_param(param):
    if isinstance(param, bool):
        return str(param).lower()
    return param