import codecs
import hashlib
import io
import json
import os
import sys
import atexit
import shutil
import tempfile
def toJSONFilter(action):
    """Like `toJSONFilters`, but takes a single action as argument.
    """
    toJSONFilters([action])