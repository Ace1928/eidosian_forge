import email.parser
import time
from difflib import SequenceMatcher
from typing import BinaryIO, Optional, TextIO, Union
from .objects import S_ISGITLINK, Blob, Commit
from .pack import ObjectContainer
def write_object_diff(f, store: ObjectContainer, old_file, new_file, diff_binary=False):
    """Write the diff for an object.

    Args:
      f: File-like object to write to
      store: Store to retrieve objects from, if necessary
      old_file: (path, mode, hexsha) tuple
      new_file: (path, mode, hexsha) tuple
      diff_binary: Whether to diff files even if they
        are considered binary files by is_binary().

    Note: the tuple elements should be None for nonexistent files
    """
    old_path, old_mode, old_id = old_file
    new_path, new_mode, new_id = new_file
    patched_old_path = patch_filename(old_path, b'a')
    patched_new_path = patch_filename(new_path, b'b')

    def content(mode, hexsha):
        if hexsha is None:
            return Blob.from_string(b'')
        elif S_ISGITLINK(mode):
            return Blob.from_string(b'Subproject commit ' + hexsha + b'\n')
        else:
            return store[hexsha]

    def lines(content):
        if not content:
            return []
        else:
            return content.splitlines()
    f.writelines(gen_diff_header((old_path, new_path), (old_mode, new_mode), (old_id, new_id)))
    old_content = content(old_mode, old_id)
    new_content = content(new_mode, new_id)
    if not diff_binary and (is_binary(old_content.data) or is_binary(new_content.data)):
        binary_diff = b'Binary files ' + patched_old_path + b' and ' + patched_new_path + b' differ\n'
        f.write(binary_diff)
    else:
        f.writelines(unified_diff(lines(old_content), lines(new_content), patched_old_path, patched_new_path))