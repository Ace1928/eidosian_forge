import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _GetStandaloneExecutableSuffix(self):
    if 'product_extension' in self.spec:
        return '.' + self.spec['product_extension']
    return {'executable': '', 'static_library': '.a', 'shared_library': '.dylib', 'loadable_module': '.so'}[self.spec['type']]