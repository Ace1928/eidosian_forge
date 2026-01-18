import io
import os
import struct
import sys
def whichdb(filename):
    """Guess which db package to use to open a db file.

    Return values:

    - None if the database file can't be read;
    - empty string if the file can be read but can't be recognized
    - the name of the dbm submodule (e.g. "ndbm" or "gnu") if recognized.

    Importing the given module may still fail, and opening the
    database using that module may still fail.
    """
    filename = os.fsencode(filename)
    try:
        f = io.open(filename + b'.pag', 'rb')
        f.close()
        f = io.open(filename + b'.dir', 'rb')
        f.close()
        return 'dbm.ndbm'
    except OSError:
        try:
            f = io.open(filename + b'.db', 'rb')
            f.close()
            if ndbm is not None:
                d = ndbm.open(filename)
                d.close()
                return 'dbm.ndbm'
        except OSError:
            pass
    try:
        os.stat(filename + b'.dat')
        size = os.stat(filename + b'.dir').st_size
        if size == 0:
            return 'dbm.dumb'
        f = io.open(filename + b'.dir', 'rb')
        try:
            if f.read(1) in (b"'", b'"'):
                return 'dbm.dumb'
        finally:
            f.close()
    except OSError:
        pass
    try:
        f = io.open(filename, 'rb')
    except OSError:
        return None
    with f:
        s16 = f.read(16)
    s = s16[0:4]
    if len(s) != 4:
        return ''
    try:
        magic, = struct.unpack('=l', s)
    except struct.error:
        return ''
    if magic in (324508366, 324508365, 324508367):
        return 'dbm.gnu'
    try:
        magic, = struct.unpack('=l', s16[-4:])
    except struct.error:
        return ''
    return ''