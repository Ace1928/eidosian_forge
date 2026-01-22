import sys
import os
import re
import warnings
from .errors import (
from .spawn import spawn
from .file_util import move_file
from .dir_util import mkpath
from ._modified import newer_group
from .util import split_quoted, execute
from ._log import log
def _setup_compile(self, outdir, macros, incdirs, sources, depends, extra):
    """Process arguments and decide which source files to compile."""
    outdir, macros, incdirs = self._fix_compile_args(outdir, macros, incdirs)
    if extra is None:
        extra = []
    objects = self.object_filenames(sources, strip_dir=0, output_dir=outdir)
    assert len(objects) == len(sources)
    pp_opts = gen_preprocess_options(macros, incdirs)
    build = {}
    for i in range(len(sources)):
        src = sources[i]
        obj = objects[i]
        ext = os.path.splitext(src)[1]
        self.mkpath(os.path.dirname(obj))
        build[obj] = (src, ext)
    return (macros, objects, extra, pp_opts, build)