from __future__ import nested_scopes
import fnmatch
import os.path
from _pydev_runfiles.pydev_runfiles_coverage import start_coverage_support
from _pydevd_bundle.pydevd_constants import *  # @UnusedWildImport
import re
import time
def __get_module_from_str(self, modname, print_exception, pyfile):
    """ Import the module in the given import path.
            * Returns the "final" module, so importing "coilib40.subject.visu"
            returns the "visu" module, not the "coilib40" as returned by __import__ """
    try:
        mod = __import__(modname)
        for part in modname.split('.')[1:]:
            mod = getattr(mod, part)
        return mod
    except:
        if print_exception:
            from _pydev_runfiles import pydev_runfiles_xml_rpc
            from _pydevd_bundle import pydevd_io
            buf_err = pydevd_io.start_redirect(keep_original_redirection=True, std='stderr')
            buf_out = pydevd_io.start_redirect(keep_original_redirection=True, std='stdout')
            try:
                import traceback
                traceback.print_exc()
                sys.stderr.write('ERROR: Module: %s could not be imported (file: %s).\n' % (modname, pyfile))
            finally:
                pydevd_io.end_redirect('stderr')
                pydevd_io.end_redirect('stdout')
            pydev_runfiles_xml_rpc.notifyTest('error', buf_out.getvalue(), buf_err.getvalue(), pyfile, modname, 0)
        return None