import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def BuildCygwinBashCommandLine(self, args, path_to_base):
    """Build a command line that runs args via cygwin bash. We assume that all
        incoming paths are in Windows normpath'd form, so they need to be
        converted to posix style for the part of the command line that's passed to
        bash. We also have to do some Visual Studio macro emulation here because
        various rules use magic VS names for things. Also note that rules that
        contain ninja variables cannot be fixed here (for example ${source}), so
        the outer generator needs to make sure that the paths that are written out
        are in posix style, if the command line will be used here."""
    cygwin_dir = os.path.normpath(os.path.join(path_to_base, self.msvs_cygwin_dirs[0]))
    cd = ('cd %s' % path_to_base).replace('\\', '/')
    args = [a.replace('\\', '/').replace('"', '\\"') for a in args]
    args = ["'%s'" % a.replace("'", "'\\''") for a in args]
    bash_cmd = ' '.join(args)
    cmd = 'call "%s\\setup_env.bat" && set CYGWIN=nontsec && ' % cygwin_dir + f'bash -c "{cd} ; {bash_cmd}"'
    return cmd