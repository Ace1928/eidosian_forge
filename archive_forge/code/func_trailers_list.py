import datetime
import re
from subprocess import Popen, PIPE
from gitdb import IStream
from git.util import hex_to_bin, Actor, Stats, finalize_process
from git.diff import Diffable
from git.cmd import Git
from .tree import Tree
from . import base
from .util import (
from time import time, daylight, altzone, timezone, localtime
import os
from io import BytesIO
import logging
from collections import defaultdict
from typing import (
from git.types import PathLike, Literal
@property
def trailers_list(self) -> List[Tuple[str, str]]:
    """Get the trailers of the message as a list.

        Git messages can contain trailer information that are similar to RFC 822
        e-mail headers (see: https://git-scm.com/docs/git-interpret-trailers).

        This functions calls ``git interpret-trailers --parse`` onto the message
        to extract the trailer information, returns the raw trailer data as a list.

        Valid message with trailer::

            Subject line

            some body information

            another information

            key1: value1.1
            key1: value1.2
            key2 :    value 2 with inner spaces


        Returned list will look like this::

            [
                ("key1", "value1.1"),
                ("key1", "value1.2"),
                ("key2", "value 2 with inner spaces"),
            ]


        :return:
            List containing key-value tuples of whitespace stripped trailer information.
        """
    cmd = ['git', 'interpret-trailers', '--parse']
    proc: Git.AutoInterrupt = self.repo.git.execute(cmd, as_process=True, istream=PIPE)
    trailer: str = proc.communicate(str(self.message).encode())[0].decode('utf8')
    trailer = trailer.strip()
    if not trailer:
        return []
    trailer_list = []
    for t in trailer.split('\n'):
        key, val = t.split(':', 1)
        trailer_list.append((key.strip(), val.strip()))
    return trailer_list