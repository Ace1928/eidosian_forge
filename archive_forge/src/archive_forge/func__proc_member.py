from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
def _proc_member(self, tarfile):
    """Choose the right processing method depending on
           the type and call it.
        """
    if self.type in (GNUTYPE_LONGNAME, GNUTYPE_LONGLINK):
        return self._proc_gnulong(tarfile)
    elif self.type == GNUTYPE_SPARSE:
        return self._proc_sparse(tarfile)
    elif self.type in (XHDTYPE, XGLTYPE, SOLARIS_XHDTYPE):
        return self._proc_pax(tarfile)
    else:
        return self._proc_builtin(tarfile)