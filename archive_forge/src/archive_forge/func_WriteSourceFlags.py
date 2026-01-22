import gyp
import gyp.common
import gyp.generator.make as make  # Reuse global functions from make backend.
import os
import re
import subprocess
def WriteSourceFlags(self, spec, configs):
    """Write out the flags and include paths used to compile source files for
        the current target.

        Args:
          spec, configs: input from gyp.
        """
    for configname, config in sorted(configs.items()):
        extracted_includes = []
        self.WriteLn('\n# Flags passed to both C and C++ files.')
        cflags, includes_from_cflags = self.ExtractIncludesFromCFlags(config.get('cflags', []) + config.get('cflags_c', []))
        extracted_includes.extend(includes_from_cflags)
        self.WriteList(cflags, 'MY_CFLAGS_%s' % configname)
        self.WriteList(config.get('defines'), 'MY_DEFS_%s' % configname, prefix='-D', quoter=make.EscapeCppDefine)
        self.WriteLn('\n# Include paths placed before CFLAGS/CPPFLAGS')
        includes = list(config.get('include_dirs', []))
        includes.extend(extracted_includes)
        includes = map(Sourceify, map(self.LocalPathify, includes))
        includes = self.NormalizeIncludePaths(includes)
        self.WriteList(includes, 'LOCAL_C_INCLUDES_%s' % configname)
        self.WriteLn('\n# Flags passed to only C++ (and not C) files.')
        self.WriteList(config.get('cflags_cc'), 'LOCAL_CPPFLAGS_%s' % configname)
    self.WriteLn('\nLOCAL_CFLAGS := $(MY_CFLAGS_$(GYP_CONFIGURATION)) $(MY_DEFS_$(GYP_CONFIGURATION))')
    if self.toolset == 'host':
        self.WriteLn('# Undefine ANDROID for host modules')
        self.WriteLn('LOCAL_CFLAGS += -UANDROID')
    self.WriteLn('LOCAL_C_INCLUDES := $(GYP_COPIED_SOURCE_ORIGIN_DIRS) $(LOCAL_C_INCLUDES_$(GYP_CONFIGURATION))')
    self.WriteLn('LOCAL_CPPFLAGS := $(LOCAL_CPPFLAGS_$(GYP_CONFIGURATION))')
    self.WriteLn('LOCAL_ASFLAGS := $(LOCAL_CFLAGS)')