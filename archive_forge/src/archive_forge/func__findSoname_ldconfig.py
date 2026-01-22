import os
import shutil
import subprocess
import sys
def _findSoname_ldconfig(name):
    import struct
    if struct.calcsize('l') == 4:
        machine = os.uname().machine + '-32'
    else:
        machine = os.uname().machine + '-64'
    mach_map = {'x86_64-64': 'libc6,x86-64', 'ppc64-64': 'libc6,64bit', 'sparc64-64': 'libc6,64bit', 's390x-64': 'libc6,64bit', 'ia64-64': 'libc6,IA-64'}
    abi_type = mach_map.get(machine, 'libc6')
    regex = '\\s+(lib%s\\.[^\\s]+)\\s+\\(%s'
    regex = os.fsencode(regex % (re.escape(name), abi_type))
    try:
        with subprocess.Popen(['/sbin/ldconfig', '-p'], stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE, env={'LC_ALL': 'C', 'LANG': 'C'}) as p:
            res = re.search(regex, p.stdout.read())
            if res:
                return os.fsdecode(res.group(1))
    except OSError:
        pass