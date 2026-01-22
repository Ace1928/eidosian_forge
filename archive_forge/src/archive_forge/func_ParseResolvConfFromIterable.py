import socket, string, types, time, select
import errno
from . import Type,Class,Opcode
from . import Lib
def ParseResolvConfFromIterable(lines):
    """parses a resolv.conf formatted stream and sets defaults for name servers"""
    global defaults
    for line in lines:
        line = line.strip()
        if not line or line[0] == ';' or line[0] == '#':
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        if fields[0] == 'domain' and len(fields) > 1:
            defaults['domain'] = fields[1]
        if fields[0] == 'search':
            pass
        if fields[0] == 'options':
            pass
        if fields[0] == 'sortlist':
            pass
        if fields[0] == 'nameserver':
            defaults['server'].append(fields[1])