import collections
import copy
import hashlib
import json
import multiprocessing
import os.path
import re
import signal
import subprocess
import sys
import gyp
import gyp.common
import gyp.msvs_emulation
import gyp.MSVSUtil as MSVSUtil
import gyp.xcode_emulation
from io import StringIO
from gyp.common import GetEnvironFallback
import gyp.ninja_syntax as ninja_syntax
def WriteMacBundle(self, spec, mac_bundle_depends, is_empty):
    assert self.is_mac_bundle
    package_framework = spec['type'] in ('shared_library', 'loadable_module')
    output = self.ComputeMacBundleOutput()
    if is_empty:
        output += '.stamp'
    variables = []
    self.AppendPostbuildVariable(variables, spec, output, self.target.binary, is_command_start=not package_framework)
    if package_framework and (not is_empty):
        if spec['type'] == 'shared_library' and self.xcode_settings.isIOS:
            self.ninja.build(output, 'package_ios_framework', mac_bundle_depends, variables=variables)
        else:
            variables.append(('version', self.xcode_settings.GetFrameworkVersion()))
            self.ninja.build(output, 'package_framework', mac_bundle_depends, variables=variables)
    else:
        self.ninja.build(output, 'stamp', mac_bundle_depends, variables=variables)
    self.target.bundle = output
    return output