import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def GetGlobalVSMacroEnv(vs_version):
    """Get a dict of variables mapping internal VS macro names to their gyp
    equivalents. Returns all variables that are independent of the target."""
    env = {}
    if vs_version.Path():
        env['$(VSInstallDir)'] = vs_version.Path()
        env['$(VCInstallDir)'] = os.path.join(vs_version.Path(), 'VC') + '\\'
    dxsdk_dir = _FindDirectXInstallation()
    env['$(DXSDK_DIR)'] = dxsdk_dir if dxsdk_dir else ''
    env['$(WDK_DIR)'] = os.environ.get('WDK_DIR', '')
    return env