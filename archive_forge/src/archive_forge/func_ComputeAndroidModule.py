import gyp
import gyp.common
import gyp.generator.make as make  # Reuse global functions from make backend.
import os
import re
import subprocess
def ComputeAndroidModule(self, spec):
    """Return the Android module name used for a gyp spec.

        We use the complete qualified target name to avoid collisions between
        duplicate targets in different directories. We also add a suffix to
        distinguish gyp-generated module names.
        """
    if int(spec.get('android_unmangled_name', 0)):
        assert self.type != 'shared_library' or self.target.startswith('lib')
        return self.target
    if self.type == 'shared_library':
        prefix = 'lib_'
    else:
        prefix = ''
    if spec['toolset'] == 'host':
        suffix = '_$(TARGET_$(GYP_VAR_PREFIX)ARCH)_host_gyp'
    else:
        suffix = '_gyp'
    if self.path:
        middle = make.StringToMakefileVariable(f'{self.path}_{self.target}')
    else:
        middle = make.StringToMakefileVariable(self.target)
    return ''.join([prefix, middle, suffix])