import os
import re
import shutil
import subprocess
import stat
import string
import sys
def _UseSeparateMspdbsrv(self, env, args):
    """Allows to use a unique instance of mspdbsrv.exe per linker instead of a
    shared one."""
    if len(args) < 1:
        raise Exception('Not enough arguments')
    if args[0] != 'link.exe':
        return
    endpoint_name = None
    for arg in args:
        m = _LINK_EXE_OUT_ARG.match(arg)
        if m:
            endpoint_name = re.sub('\\W+', '', '%s_%d' % (m.group('out'), os.getpid()))
            break
    if endpoint_name is None:
        return
    env['_MSPDBSRV_ENDPOINT_'] = endpoint_name