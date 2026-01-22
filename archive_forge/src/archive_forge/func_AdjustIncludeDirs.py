import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def AdjustIncludeDirs(self, include_dirs, config):
    """Updates include_dirs to expand VS specific paths, and adds the system
        include dirs used for platform SDK and similar."""
    config = self._TargetConfig(config)
    includes = include_dirs + self.msvs_system_include_dirs[config]
    includes.extend(self._Setting(('VCCLCompilerTool', 'AdditionalIncludeDirectories'), config, default=[]))
    return [self.ConvertVSMacros(p, config=config) for p in includes]