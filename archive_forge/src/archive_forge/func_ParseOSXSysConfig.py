import socket, string, types, time, select
import errno
from . import Type,Class,Opcode
from . import Lib
def ParseOSXSysConfig():
    """Retrieves the current Mac OS X resolver settings using the scutil(8) command."""
    import os, re
    scutil = os.popen('/usr/sbin/scutil --dns', 'r')
    res_re = re.compile('^\\s+nameserver[]0-9[]*\\s*\\:\\s*(\\S+)$')
    sets = []
    currentset = None
    while True:
        l = scutil.readline()
        if not l:
            break
        l = l.rstrip()
        if len(l) < 1 or l[0] not in string.whitespace:
            currentset = None
            continue
        m = res_re.match(l)
        if m:
            if currentset is None:
                currentset = []
                sets.append(currentset)
            currentset.append(m.group(1))
    scutil.close()
    for currentset in sets:
        defaults['server'].extend(currentset)