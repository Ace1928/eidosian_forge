import os
import sys
import subprocess
import shlex
from distutils.errors import CompileError, DistutilsExecError, LibError
from distutils.unixccompiler import UnixCCompiler
from numpy.distutils.ccompiler import replace_method
from numpy.distutils.misc_util import _commandline_dep_string
from numpy.distutils import log
def UnixCCompiler_create_static_lib(self, objects, output_libname, output_dir=None, debug=0, target_lang=None):
    """
    Build a static library in a separate sub-process.

    Parameters
    ----------
    objects : list or tuple of str
        List of paths to object files used to build the static library.
    output_libname : str
        The library name as an absolute or relative (if `output_dir` is used)
        path.
    output_dir : str, optional
        The path to the output directory. Default is None, in which case
        the ``output_dir`` attribute of the UnixCCompiler instance.
    debug : bool, optional
        This parameter is not used.
    target_lang : str, optional
        This parameter is not used.

    Returns
    -------
    None

    """
    objects, output_dir = self._fix_object_args(objects, output_dir)
    output_filename = self.library_filename(output_libname, output_dir=output_dir)
    if self._need_link(objects, output_filename):
        try:
            os.unlink(output_filename)
        except OSError:
            pass
        self.mkpath(os.path.dirname(output_filename))
        tmp_objects = objects + self.objects
        while tmp_objects:
            objects = tmp_objects[:50]
            tmp_objects = tmp_objects[50:]
            display = '%s: adding %d object files to %s' % (os.path.basename(self.archiver[0]), len(objects), output_filename)
            self.spawn(self.archiver + [output_filename] + objects, display=display)
        if self.ranlib:
            display = '%s:@ %s' % (os.path.basename(self.ranlib[0]), output_filename)
            try:
                self.spawn(self.ranlib + [output_filename], display=display)
            except DistutilsExecError as e:
                msg = str(e)
                raise LibError(msg) from None
    else:
        log.debug('skipping %s (up-to-date)', output_filename)
    return