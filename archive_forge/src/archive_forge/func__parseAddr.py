from paste.util import intset
import socket
def _parseAddr(addr, lookup=True):
    if lookup and any((ch not in IP4Range._IPREMOVE for ch in addr)):
        try:
            addr = socket.gethostbyname(addr)
        except socket.error:
            raise ValueError('Invalid Hostname as argument.')
    naddr = 0
    for naddrpos, part in enumerate(addr.split('.')):
        if naddrpos >= 4:
            raise ValueError('Address contains more than four parts.')
        try:
            if not part:
                part = 0
            else:
                part = int(part)
            if not 0 <= part < 256:
                raise ValueError
        except ValueError:
            raise ValueError('Address part out of range.')
        naddr <<= 8
        naddr += part
    return (naddr, naddrpos + 1)