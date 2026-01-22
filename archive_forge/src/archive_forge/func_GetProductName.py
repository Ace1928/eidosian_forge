import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetProductName(self):
    """Returns PRODUCT_NAME."""
    return self.spec.get('product_name', self.spec['target_name'])