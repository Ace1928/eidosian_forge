import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def _GetDefFileAsLdflags(self, ldflags, gyp_to_build_path):
    """.def files get implicitly converted to a ModuleDefinitionFile for the
        linker in the VS generator. Emulate that behaviour here."""
    def_file = self.GetDefFile(gyp_to_build_path)
    if def_file:
        ldflags.append('/DEF:"%s"' % def_file)