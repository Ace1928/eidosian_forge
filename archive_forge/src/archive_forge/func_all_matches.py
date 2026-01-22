import os
import re
import stat
import sys
import fnmatch
from xdg import BaseDirectory
import xdg.Locale
from xml.dom import minidom, XML_NAMESPACE
from collections import defaultdict
def all_matches(self, path):
    """Return a list of (MIMEtype, glob weight) pairs for the path."""
    return list(self._match_path(path))