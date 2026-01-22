import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _GetIOSCodeSignIdentityKey(self, settings):
    identity = settings.get('CODE_SIGN_IDENTITY')
    if not identity:
        return None
    if identity not in XcodeSettings._codesigning_key_cache:
        output = subprocess.check_output(['security', 'find-identity', '-p', 'codesigning', '-v'])
        for line in output.splitlines():
            if identity in line:
                fingerprint = line.split()[1]
                cache = XcodeSettings._codesigning_key_cache
                assert identity not in cache or fingerprint == cache[identity], 'Multiple codesigning fingerprints for identity: %s' % identity
                XcodeSettings._codesigning_key_cache[identity] = fingerprint
    return XcodeSettings._codesigning_key_cache.get(identity, '')