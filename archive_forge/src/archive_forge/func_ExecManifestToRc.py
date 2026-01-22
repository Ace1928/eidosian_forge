import os
import re
import shutil
import subprocess
import stat
import string
import sys
def ExecManifestToRc(self, arch, *args):
    """Creates a resource file pointing a SxS assembly manifest.
    |args| is tuple containing path to resource file, path to manifest file
    and resource name which can be "1" (for executables) or "2" (for DLLs)."""
    manifest_path, resource_path, resource_name = args
    with open(resource_path, 'w') as output:
        output.write('#include <windows.h>\n%s RT_MANIFEST "%s"' % (resource_name, os.path.abspath(manifest_path).replace('\\', '/')))