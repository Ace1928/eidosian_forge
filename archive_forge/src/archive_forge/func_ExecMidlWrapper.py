import os
import re
import shutil
import subprocess
import stat
import string
import sys
def ExecMidlWrapper(self, arch, outdir, tlb, h, dlldata, iid, proxy, idl, *flags):
    """Filter noisy filenames output from MIDL compile step that isn't
    quietable via command line flags.
    """
    args = ['midl', '/nologo'] + list(flags) + ['/out', outdir, '/tlb', tlb, '/h', h, '/dlldata', dlldata, '/iid', iid, '/proxy', proxy, idl]
    env = self._GetEnv(arch)
    popen = subprocess.Popen(args, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = popen.communicate()[0].decode('utf-8')
    lines = out.splitlines()
    prefixes = ('Processing ', '64 bit Processing ')
    processing = {os.path.basename(x) for x in lines if x.startswith(prefixes)}
    for line in lines:
        if not line.startswith(prefixes) and line not in processing:
            print(line)
    return popen.returncode