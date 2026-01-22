import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def GetDefFile(self, gyp_to_build_path):
    """Returns the .def file from sources, if any.  Otherwise returns None."""
    spec = self.spec
    if spec['type'] in ('shared_library', 'loadable_module', 'executable'):
        def_files = [s for s in spec.get('sources', []) if s.lower().endswith('.def')]
        if len(def_files) == 1:
            return gyp_to_build_path(def_files[0])
        elif len(def_files) > 1:
            raise Exception('Multiple .def files')
    return None