import email.parser
import time
from difflib import SequenceMatcher
from typing import BinaryIO, Optional, TextIO, Union
from .objects import S_ISGITLINK, Blob, Commit
from .pack import ObjectContainer
def write_blob_diff(f, old_file, new_file):
    """Write blob diff.

    Args:
      f: File-like object to write to
      old_file: (path, mode, hexsha) tuple (None if nonexisting)
      new_file: (path, mode, hexsha) tuple (None if nonexisting)

    Note: The use of write_object_diff is recommended over this function.
    """
    old_path, old_mode, old_blob = old_file
    new_path, new_mode, new_blob = new_file
    patched_old_path = patch_filename(old_path, b'a')
    patched_new_path = patch_filename(new_path, b'b')

    def lines(blob):
        if blob is not None:
            return blob.splitlines()
        else:
            return []
    f.writelines(gen_diff_header((old_path, new_path), (old_mode, new_mode), (getattr(old_blob, 'id', None), getattr(new_blob, 'id', None))))
    old_contents = lines(old_blob)
    new_contents = lines(new_blob)
    f.writelines(unified_diff(old_contents, new_contents, patched_old_path, patched_new_path))