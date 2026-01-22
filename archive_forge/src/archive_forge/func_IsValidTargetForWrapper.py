import errno
import gyp.generator.ninja
import os
import re
import xml.sax.saxutils
def IsValidTargetForWrapper(target_extras, executable_target_pattern, spec):
    """Limit targets for Xcode wrapper.

  Xcode sometimes performs poorly with too many targets, so only include
  proper executable targets, with filters to customize.
  Arguments:
    target_extras: Regular expression to always add, matching any target.
    executable_target_pattern: Regular expression limiting executable targets.
    spec: Specifications for target.
  """
    target_name = spec.get('target_name')
    if target_extras is not None and re.search(target_extras, target_name):
        return True
    if int(spec.get('mac_xctest_bundle', 0)) != 0 or (spec.get('type', '') == 'executable' and spec.get('product_extension', '') != 'bundle'):
        if executable_target_pattern is not None:
            if not re.search(executable_target_pattern, target_name):
                return False
        return True
    return False