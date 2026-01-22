import re
import sys
from typing import List, Optional, Tuple
def _parse_patch(lines: List[bytes]) -> Tuple[List[bytes], List[bool], List[Tuple[int, int]]]:
    """Parse a git style diff or patch to generate diff stats.

    Args:
      lines: list of byte string lines from the diff to be parsed
    Returns: A tuple (names, is_binary, counts) of three lists
    """
    names = []
    nametypes = []
    counts = []
    in_patch_chunk = in_git_header = binaryfile = False
    currentfile: Optional[bytes] = None
    added = deleted = 0
    for line in lines:
        if line.startswith(_GIT_HEADER_START):
            if currentfile is not None:
                names.append(currentfile)
                nametypes.append(binaryfile)
                counts.append((added, deleted))
            m = _git_header_name.search(line)
            assert m
            currentfile = m.group(2)
            binaryfile = False
            added = deleted = 0
            in_git_header = True
            in_patch_chunk = False
        elif line.startswith(_GIT_BINARY_START) and in_git_header:
            binaryfile = True
            in_git_header = False
        elif line.startswith(_GIT_RENAMEFROM_START) and in_git_header:
            currentfile = line[12:]
        elif line.startswith(_GIT_RENAMETO_START) and in_git_header:
            assert currentfile
            currentfile += b' => %s' % line[10:]
        elif line.startswith(_GIT_CHUNK_START) and (in_patch_chunk or in_git_header):
            in_patch_chunk = True
            in_git_header = False
        elif line.startswith(_GIT_ADDED_START) and in_patch_chunk:
            added += 1
        elif line.startswith(_GIT_DELETED_START) and in_patch_chunk:
            deleted += 1
        elif not line.startswith(_GIT_UNCHANGED_START) and in_patch_chunk:
            in_patch_chunk = False
    if currentfile is not None:
        names.append(currentfile)
        nametypes.append(binaryfile)
        counts.append((added, deleted))
    return (names, nametypes, counts)