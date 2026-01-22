import fcntl
import fnmatch
import glob
import json
import os
import plistlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile
def _CopyXIBFile(self, source, dest):
    """Compiles a XIB file with ibtool into a binary plist in the bundle."""
    base = os.path.dirname(os.path.realpath(__file__))
    if os.path.relpath(source):
        source = os.path.join(base, source)
    if os.path.relpath(dest):
        dest = os.path.join(base, dest)
    args = ['xcrun', 'ibtool', '--errors', '--warnings', '--notices']
    if os.environ['XCODE_VERSION_ACTUAL'] > '0700':
        args.extend(['--auto-activate-custom-fonts'])
        if 'IPHONEOS_DEPLOYMENT_TARGET' in os.environ:
            args.extend(['--target-device', 'iphone', '--target-device', 'ipad', '--minimum-deployment-target', os.environ['IPHONEOS_DEPLOYMENT_TARGET']])
        else:
            args.extend(['--target-device', 'mac', '--minimum-deployment-target', os.environ['MACOSX_DEPLOYMENT_TARGET']])
    args.extend(['--output-format', 'human-readable-text', '--compile', dest, source])
    ibtool_section_re = re.compile('/\\*.*\\*/')
    ibtool_re = re.compile('.*note:.*is clipping its content')
    try:
        stdout = subprocess.check_output(args)
    except subprocess.CalledProcessError as e:
        print(e.output)
        raise
    current_section_header = None
    for line in stdout.splitlines():
        if ibtool_section_re.match(line):
            current_section_header = line
        elif not ibtool_re.match(line):
            if current_section_header:
                print(current_section_header)
                current_section_header = None
            print(line)
    return 0