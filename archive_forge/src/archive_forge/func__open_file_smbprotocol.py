from __future__ import absolute_import, print_function, division
import logging
import sys
from contextlib import contextmanager
from petl.compat import PY3
from petl.io.sources import register_reader, register_writer, get_reader, get_writer
def _open_file_smbprotocol(url, mode='rb', **kwargs):
    _domain, host, port, user, passwd, server_path = _parse_smb_url(url)
    import smbclient
    try:
        if user:
            smbclient.register_session(host, username=user, password=passwd, port=port)
        mode2 = mode[:1] + 'b'
        filehandle = smbclient.open_file(server_path, mode=mode2, **kwargs)
        return filehandle
    except Exception as ex:
        raise ConnectionError('SMB error: %s' % ex).with_traceback(sys.exc_info()[2])