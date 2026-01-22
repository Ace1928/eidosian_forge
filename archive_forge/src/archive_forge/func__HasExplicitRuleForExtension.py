import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def _HasExplicitRuleForExtension(self, spec, extension):
    """Determine if there's an explicit rule for a particular extension."""
    for rule in spec.get('rules', []):
        if rule['extension'] == extension:
            return True
    return False