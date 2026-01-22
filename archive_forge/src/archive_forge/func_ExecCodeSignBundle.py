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
def ExecCodeSignBundle(self, key, entitlements, provisioning, path, preserve):
    """Code sign a bundle.

    This function tries to code sign an iOS bundle, following the same
    algorithm as Xcode:
      1. pick the provisioning profile that best match the bundle identifier,
         and copy it into the bundle as embedded.mobileprovision,
      2. copy Entitlements.plist from user or SDK next to the bundle,
      3. code sign the bundle.
    """
    substitutions, overrides = self._InstallProvisioningProfile(provisioning, self._GetCFBundleIdentifier())
    entitlements_path = self._InstallEntitlements(entitlements, substitutions, overrides)
    args = ['codesign', '--force', '--sign', key]
    if preserve == 'True':
        args.extend(['--deep', '--preserve-metadata=identifier,entitlements'])
    else:
        args.extend(['--entitlements', entitlements_path])
    args.extend(['--timestamp=none', path])
    subprocess.check_call(args)