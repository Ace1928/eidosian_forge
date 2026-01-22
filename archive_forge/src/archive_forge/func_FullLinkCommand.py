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
def FullLinkCommand(ldcmd, out, binary_type):
    resource_name = {'exe': '1', 'dll': '2'}[binary_type]
    return '%(python)s gyp-win-tool link-with-manifests $arch %(embed)s %(out)s "%(ldcmd)s" %(resname)s $mt $rc "$intermediatemanifest" $manifests' % {'python': sys.executable, 'out': out, 'ldcmd': ldcmd, 'resname': resource_name, 'embed': embed_manifest}