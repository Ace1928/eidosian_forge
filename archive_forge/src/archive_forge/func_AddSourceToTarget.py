import filecmp
import gyp.common
import gyp.xcodeproj_file
import gyp.xcode_ninja
import errno
import os
import sys
import posixpath
import re
import shutil
import subprocess
import tempfile
def AddSourceToTarget(source, type, pbxp, xct):
    source_extensions = ['c', 'cc', 'cpp', 'cxx', 'm', 'mm', 's', 'swift']
    library_extensions = ['a', 'dylib', 'framework', 'o']
    basename = posixpath.basename(source)
    root, ext = posixpath.splitext(basename)
    if ext:
        ext = ext[1:].lower()
    if ext in source_extensions and type != 'none':
        xct.SourcesPhase().AddFile(source)
    elif ext in library_extensions and type != 'none':
        xct.FrameworksPhase().AddFile(source)
    else:
        pbxp.AddOrGetFileInRootGroup(source)