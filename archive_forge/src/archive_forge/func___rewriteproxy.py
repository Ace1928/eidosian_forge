import base64
import socket
import struct
import sys
def __rewriteproxy(self, header):
    """ rewrite HTTP request headers to support non-tunneling proxies
        (i.e. those which do not support the CONNECT method).
        This only works for HTTP (not HTTPS) since HTTPS requires tunneling.
        """
    host, endpt = (None, None)
    hdrs = header.split('\r\n')
    for hdr in hdrs:
        if hdr.lower().startswith('host:'):
            host = hdr
        elif hdr.lower().startswith('get') or hdr.lower().startswith('post'):
            endpt = hdr
    if host and endpt:
        hdrs.remove(host)
        hdrs.remove(endpt)
        host = host.split(' ')[1]
        endpt = endpt.split(' ')
        if self.__proxy[4] != None and self.__proxy[5] != None:
            hdrs.insert(0, self.__getauthheader())
        hdrs.insert(0, 'Host: %s' % host)
        hdrs.insert(0, '%s http://%s%s %s' % (endpt[0], host, endpt[1], endpt[2]))
    return '\r\n'.join(hdrs)