import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetCflags(self, configname, arch=None):
    """Returns flags that need to be added to .c, .cc, .m, and .mm
    compilations."""
    self.configname = configname
    cflags = []
    sdk_root = self._SdkPath()
    if 'SDKROOT' in self._Settings() and sdk_root:
        cflags.append('-isysroot %s' % sdk_root)
    if self.header_map_path:
        cflags.append('-I%s' % self.header_map_path)
    if self._Test('CLANG_WARN_CONSTANT_CONVERSION', 'YES', default='NO'):
        cflags.append('-Wconstant-conversion')
    if self._Test('GCC_CHAR_IS_UNSIGNED_CHAR', 'YES', default='NO'):
        cflags.append('-funsigned-char')
    if self._Test('GCC_CW_ASM_SYNTAX', 'YES', default='YES'):
        cflags.append('-fasm-blocks')
    if 'GCC_DYNAMIC_NO_PIC' in self._Settings():
        if self._Settings()['GCC_DYNAMIC_NO_PIC'] == 'YES':
            cflags.append('-mdynamic-no-pic')
    else:
        pass
    if self._Test('GCC_ENABLE_PASCAL_STRINGS', 'YES', default='YES'):
        cflags.append('-mpascal-strings')
    self._Appendf(cflags, 'GCC_OPTIMIZATION_LEVEL', '-O%s', default='s')
    if self._Test('GCC_GENERATE_DEBUGGING_SYMBOLS', 'YES', default='YES'):
        dbg_format = self._Settings().get('DEBUG_INFORMATION_FORMAT', 'dwarf')
        if dbg_format == 'dwarf':
            cflags.append('-gdwarf-2')
        elif dbg_format == 'stabs':
            raise NotImplementedError('stabs debug format is not supported yet.')
        elif dbg_format == 'dwarf-with-dsym':
            cflags.append('-gdwarf-2')
        else:
            raise NotImplementedError('Unknown debug format %s' % dbg_format)
    if self._Settings().get('GCC_STRICT_ALIASING') == 'YES':
        cflags.append('-fstrict-aliasing')
    elif self._Settings().get('GCC_STRICT_ALIASING') == 'NO':
        cflags.append('-fno-strict-aliasing')
    if self._Test('GCC_SYMBOLS_PRIVATE_EXTERN', 'YES', default='NO'):
        cflags.append('-fvisibility=hidden')
    if self._Test('GCC_TREAT_WARNINGS_AS_ERRORS', 'YES', default='NO'):
        cflags.append('-Werror')
    if self._Test('GCC_WARN_ABOUT_MISSING_NEWLINE', 'YES', default='NO'):
        cflags.append('-Wnewline-eof')
    if self._Test('LLVM_LTO', 'YES', default='NO'):
        cflags.append('-flto')
    self._AppendPlatformVersionMinFlags(cflags)
    if self._Test('COPY_PHASE_STRIP', 'YES', default='NO'):
        self._WarnUnimplemented('COPY_PHASE_STRIP')
    self._WarnUnimplemented('GCC_DEBUGGING_SYMBOLS')
    self._WarnUnimplemented('GCC_ENABLE_OBJC_EXCEPTIONS')
    self._WarnUnimplemented('MACH_O_TYPE')
    self._WarnUnimplemented('PRODUCT_TYPE')
    if not gyp.common.CrossCompileRequested():
        if arch is not None:
            archs = [arch]
        else:
            assert self.configname
            archs = self.GetActiveArchs(self.configname)
        if len(archs) != 1:
            self._WarnUnimplemented('ARCHS')
            archs = ['i386']
        cflags.append('-arch ' + archs[0])
        if archs[0] in ('i386', 'x86_64'):
            if self._Test('GCC_ENABLE_SSE3_EXTENSIONS', 'YES', default='NO'):
                cflags.append('-msse3')
            if self._Test('GCC_ENABLE_SUPPLEMENTAL_SSE3_INSTRUCTIONS', 'YES', default='NO'):
                cflags.append('-mssse3')
            if self._Test('GCC_ENABLE_SSE41_EXTENSIONS', 'YES', default='NO'):
                cflags.append('-msse4.1')
            if self._Test('GCC_ENABLE_SSE42_EXTENSIONS', 'YES', default='NO'):
                cflags.append('-msse4.2')
    cflags += self._Settings().get('WARNING_CFLAGS', [])
    if self._IsXCTest():
        platform_root = self._XcodePlatformPath(configname)
        if platform_root:
            cflags.append('-F' + platform_root + '/Developer/Library/Frameworks/')
    if sdk_root:
        framework_root = sdk_root
    else:
        framework_root = ''
    config = self.spec['configurations'][self.configname]
    framework_dirs = config.get('mac_framework_dirs', [])
    for directory in framework_dirs:
        cflags.append('-F' + directory.replace('$(SDKROOT)', framework_root))
    self.configname = None
    return cflags