import os
import sys
import tarfile
from contextlib import closing
from io import BytesIO
from .. import errors, osutils
from ..export import _export_iter_entries
def tgz_generator(tree, dest, root, subdir, force_mtime=None, recurse_nested=False):
    """Export this tree to a new tar file.

    `dest` will be created holding the contents of this tree; if it
    already exists, it will be clobbered, like with "tar -c".
    """
    with tree.lock_read():
        import gzip
        if force_mtime is not None:
            root_mtime = force_mtime
        elif getattr(tree, 'repository', None) and getattr(tree, 'get_revision_id', None):
            rev = tree.repository.get_revision(tree.get_revision_id())
            root_mtime = rev.timestamp
        elif tree.is_versioned(''):
            root_mtime = tree.get_file_mtime('')
        else:
            root_mtime = None
        is_stdout = False
        basename = None
        basename = os.path.basename(dest)
        buf = BytesIO()
        zipstream = gzip.GzipFile(basename, 'w', fileobj=buf, mtime=root_mtime)
        for chunk in tarball_generator(tree, root, subdir, force_mtime, recurse_nested=recurse_nested):
            zipstream.write(chunk)
            yield buf.getvalue()
            buf.truncate(0)
            buf.seek(0)
        zipstream.close()
        yield buf.getvalue()