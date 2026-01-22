from __future__ import annotations
import re
import contextlib
import io
import logging
import os
import signal
from subprocess import Popen, PIPE, DEVNULL
import subprocess
import threading
from textwrap import dedent
from git.compat import defenc, force_bytes, safe_decode
from git.exc import (
from git.util import (
from typing import (
from git.types import PathLike, Literal, TBD
def _parse_object_header(self, header_line: str) -> Tuple[str, str, int]:
    """
        :param header_line:
            <hex_sha> type_string size_as_int

        :return: (hex_sha, type_string, size_as_int)

        :raise ValueError: If the header contains indication for an error due to
            incorrect input sha
        """
    tokens = header_line.split()
    if len(tokens) != 3:
        if not tokens:
            err_msg = f'SHA is empty, possible dubious ownership in the repository at {self._working_dir}.\n            If this is unintended run:\n\n                      "git config --global --add safe.directory {self._working_dir}" '
            raise ValueError(err_msg)
        else:
            raise ValueError('SHA %s could not be resolved, git returned: %r' % (tokens[0], header_line.strip()))
    if len(tokens[0]) != 40:
        raise ValueError('Failed to parse header: %r' % header_line)
    return (tokens[0], tokens[1], int(tokens[2]))