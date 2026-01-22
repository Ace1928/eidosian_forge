import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def _PchHeader(self):
    """Get the header that will appear in an #include line for all source
        files."""
    return self.settings.msvs_precompiled_header[self.config]