import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def _TargetConfig(self, config):
    """Returns the target-specific configuration."""
    if int(self.vs_version.short_name) >= 2015:
        return config
    arch = self.GetArch(config)
    if arch == 'x64' and (not config.endswith('_x64')):
        config += '_x64'
    if arch == 'x86' and config.endswith('_x64'):
        config = config.rsplit('_', 1)[0]
    return config