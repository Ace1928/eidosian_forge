import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def ComputeMacBundleOutput(self, spec):
    """Return the 'output' (full output path) to a bundle output directory."""
    assert self.is_mac_bundle
    path = generator_default_variables['PRODUCT_DIR']
    return os.path.join(path, self.xcode_settings.GetWrapperName())