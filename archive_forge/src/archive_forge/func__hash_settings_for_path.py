import errno
import glob
import hashlib
import importlib.metadata as importlib_metadata
import itertools
import json
import logging
import os
import os.path
import struct
import sys
def _hash_settings_for_path(path):
    """Return a hash and the path settings that created it."""
    paths = []
    h = hashlib.sha256()
    h.update(sys.executable.encode('utf-8'))
    h.update(sys.prefix.encode('utf-8'))
    for entry in path:
        mtime = _get_mtime(entry)
        h.update(entry.encode('utf-8'))
        h.update(_ftobytes(mtime))
        paths.append((entry, mtime))
        for ep_file in itertools.chain(glob.iglob(os.path.join(entry, '*.dist-info', 'entry_points.txt')), glob.iglob(os.path.join(entry, '*.egg-info', 'entry_points.txt'))):
            mtime = _get_mtime(ep_file)
            h.update(ep_file.encode('utf-8'))
            h.update(_ftobytes(mtime))
            paths.append((ep_file, mtime))
    return (h.hexdigest(), paths)