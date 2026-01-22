import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def WriteAndroidNdkModuleRule(self, module_name, all_sources, link_deps):
    """Write a set of LOCAL_XXX definitions for Android NDK.

        These variable definitions will be used by Android NDK but do nothing for
        non-Android applications.

        Arguments:
          module_name: Android NDK module name, which must be unique among all
              module names.
          all_sources: A list of source files (will be filtered by Compilable).
          link_deps: A list of link dependencies, which must be sorted in
              the order from dependencies to dependents.
        """
    if self.type not in ('executable', 'shared_library', 'static_library'):
        return
    self.WriteLn('# Variable definitions for Android applications')
    self.WriteLn('include $(CLEAR_VARS)')
    self.WriteLn('LOCAL_MODULE := ' + module_name)
    self.WriteLn('LOCAL_CFLAGS := $(CFLAGS_$(BUILDTYPE)) $(DEFS_$(BUILDTYPE)) $(CFLAGS_C_$(BUILDTYPE)) $(INCS_$(BUILDTYPE))')
    self.WriteLn('LOCAL_CPPFLAGS := $(CFLAGS_CC_$(BUILDTYPE))')
    self.WriteLn('LOCAL_C_INCLUDES :=')
    self.WriteLn('LOCAL_LDLIBS := $(LDFLAGS_$(BUILDTYPE)) $(LIBS)')
    cpp_ext = {'.cc': 0, '.cpp': 0, '.cxx': 0}
    default_cpp_ext = '.cpp'
    for filename in all_sources:
        ext = os.path.splitext(filename)[1]
        if ext in cpp_ext:
            cpp_ext[ext] += 1
            if cpp_ext[ext] > cpp_ext[default_cpp_ext]:
                default_cpp_ext = ext
    self.WriteLn('LOCAL_CPP_EXTENSION := ' + default_cpp_ext)
    self.WriteList(list(map(self.Absolutify, filter(Compilable, all_sources))), 'LOCAL_SRC_FILES')

    def DepsToModules(deps, prefix, suffix):
        modules = []
        for filepath in deps:
            filename = os.path.basename(filepath)
            if filename.startswith(prefix) and filename.endswith(suffix):
                modules.append(filename[len(prefix):-len(suffix)])
        return modules
    params = {'flavor': 'linux'}
    default_variables = {}
    CalculateVariables(default_variables, params)
    self.WriteList(DepsToModules(link_deps, generator_default_variables['SHARED_LIB_PREFIX'], default_variables['SHARED_LIB_SUFFIX']), 'LOCAL_SHARED_LIBRARIES')
    self.WriteList(DepsToModules(link_deps, generator_default_variables['STATIC_LIB_PREFIX'], generator_default_variables['STATIC_LIB_SUFFIX']), 'LOCAL_STATIC_LIBRARIES')
    if self.type == 'executable':
        self.WriteLn('include $(BUILD_EXECUTABLE)')
    elif self.type == 'shared_library':
        self.WriteLn('include $(BUILD_SHARED_LIBRARY)')
    elif self.type == 'static_library':
        self.WriteLn('include $(BUILD_STATIC_LIBRARY)')
    self.WriteLn()