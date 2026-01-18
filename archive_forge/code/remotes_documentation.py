from __future__ import absolute_import, print_function, division
import logging
import sys
from contextlib import contextmanager
from petl.compat import PY3
from petl.io.sources import register_reader, register_writer, get_reader, get_writer
Downloads or uploads to Windows and Samba network drives. E.g.::

        >>> def example_smb():
        ...     import petl as etl
        ...     url = 'smb://user:password@server/share/folder/file.csv'
        ...     data = b'foo,bar\na,1\nb,2\nc,2\n'
        ...     etl.tocsv(data, url)
        ...     tbl = etl.fromcsv(url)
        ...
        >>> example_smb() # doctest: +SKIP
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'a' | '1' |
        +-----+-----+
        | 'b' | '2' |
        +-----+-----+
        | 'c' | '2' |
        +-----+-----+

    The argument `url` (str) must have a URI with format:
    `smb://workgroup;user:password@server:port/share/folder/file.csv`.

    Note that you need to pass in a valid hostname or IP address for the host
    component of the URL. Do not use the Windows/NetBIOS machine name for the
    host component.

    The first component of the path in the URL points to the name of the shared
    folder. Subsequent path components will point to the directory/folder/file.

    .. note::

        For working this source require `smbprotocol`_ to be installed, e.g.::

            $ pip install smbprotocol[kerberos]

    .. versionadded:: 1.5.0

    .. _smbprotocol: https://github.com/jborean93/smbprotocol#requirements
    