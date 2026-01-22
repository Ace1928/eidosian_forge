import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def CalculateVariables(default_variables, params):
    """Calculate additional variables for use in the build (called by gyp)."""
    flavor = gyp.common.GetFlavor(params)
    if flavor == 'mac':
        default_variables.setdefault('OS', 'mac')
        default_variables.setdefault('SHARED_LIB_SUFFIX', '.dylib')
        default_variables.setdefault('SHARED_LIB_DIR', generator_default_variables['PRODUCT_DIR'])
        default_variables.setdefault('LIB_DIR', generator_default_variables['PRODUCT_DIR'])
        import gyp.generator.xcode as xcode_generator
        global generator_additional_non_configuration_keys
        generator_additional_non_configuration_keys = getattr(xcode_generator, 'generator_additional_non_configuration_keys', [])
        global generator_additional_path_sections
        generator_additional_path_sections = getattr(xcode_generator, 'generator_additional_path_sections', [])
        global generator_extra_sources_for_rules
        generator_extra_sources_for_rules = getattr(xcode_generator, 'generator_extra_sources_for_rules', [])
        COMPILABLE_EXTENSIONS.update({'.m': 'objc', '.mm': 'objcxx'})
    else:
        operating_system = flavor
        if flavor == 'android':
            operating_system = 'linux'
        default_variables.setdefault('OS', operating_system)
        if flavor == 'aix':
            default_variables.setdefault('SHARED_LIB_SUFFIX', '.a')
        else:
            default_variables.setdefault('SHARED_LIB_SUFFIX', '.so')
        default_variables.setdefault('SHARED_LIB_DIR', '$(builddir)/lib.$(TOOLSET)')
        default_variables.setdefault('LIB_DIR', '$(obj).$(TOOLSET)')