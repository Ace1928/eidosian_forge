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
def ExecCompileIosFrameworkHeaderMap(self, out, framework, *all_headers):
    framework_name = os.path.basename(framework).split('.')[0]
    all_headers = [os.path.abspath(header) for header in all_headers]
    filelist = {}
    for header in all_headers:
        filename = os.path.basename(header)
        filelist[filename] = header
        filelist[os.path.join(framework_name, filename)] = header
    WriteHmap(out, filelist)