import zipfile
import tarfile
import os
import shutil
import posixpath
import contextlib
from distutils.errors import DistutilsError
from ._path import ensure_directory
def _iter_open_tar(tar_obj, extract_dir, progress_filter):
    """Emit member-destination pairs from a tar archive."""
    tar_obj.chown = lambda *args: None
    with contextlib.closing(tar_obj):
        for member in tar_obj:
            name = member.name
            if name.startswith('/') or '..' in name.split('/'):
                continue
            prelim_dst = os.path.join(extract_dir, *name.split('/'))
            try:
                member = _resolve_tar_file_or_dir(tar_obj, member)
            except LookupError:
                continue
            final_dst = progress_filter(name, prelim_dst)
            if not final_dst:
                continue
            if final_dst.endswith(os.sep):
                final_dst = final_dst[:-1]
            yield (member, final_dst)