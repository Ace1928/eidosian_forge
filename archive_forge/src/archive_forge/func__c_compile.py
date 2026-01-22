import os
import pathlib
import subprocess
import sys
import sysconfig
import textwrap
def _c_compile(cfile, outputfilename, include_dirs=[], libraries=[], library_dirs=[]):
    if sys.platform == 'win32':
        compile_extra = ['/we4013']
        link_extra = ['/LIBPATH:' + os.path.join(sys.base_prefix, 'libs')]
    elif sys.platform.startswith('linux'):
        compile_extra = ['-O0', '-g', '-Werror=implicit-function-declaration', '-fPIC']
        link_extra = []
    else:
        compile_extra = link_extra = []
        pass
    if sys.platform == 'win32':
        link_extra = link_extra + ['/DEBUG']
    if sys.platform == 'darwin':
        for s in ('/sw/', '/opt/local/'):
            if s + 'include' not in include_dirs and os.path.exists(s + 'include'):
                include_dirs.append(s + 'include')
            if s + 'lib' not in library_dirs and os.path.exists(s + 'lib'):
                library_dirs.append(s + 'lib')
    outputfilename = outputfilename.with_suffix(get_so_suffix())
    build(cfile, outputfilename, compile_extra, link_extra, include_dirs, libraries, library_dirs)
    return outputfilename