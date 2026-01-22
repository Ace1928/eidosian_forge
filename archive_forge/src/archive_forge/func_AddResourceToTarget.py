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
def AddResourceToTarget(resource, pbxp, xct):
    xct.ResourcesPhase().AddFile(resource)