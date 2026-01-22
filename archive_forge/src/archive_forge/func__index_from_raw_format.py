import re
from git.cmd import handle_process_output
from git.compat import defenc
from git.util import finalize_process, hex_to_bin
from .objects.blob import Blob
from .objects.util import mode_str_to_int
from typing import (
from git.types import PathLike, Literal
@classmethod
def _index_from_raw_format(cls, repo: 'Repo', proc: 'Popen') -> 'DiffIndex':
    """Create a new DiffIndex from the given process output which must be in raw format.

        :param repo: The repository we are operating on
        :param proc: Process to read output from
        :return: git.DiffIndex
        """
    index: 'DiffIndex' = DiffIndex()
    handle_process_output(proc, lambda byt: cls._handle_diff_line(byt, repo, index), None, finalize_process, decode_streams=False)
    return index