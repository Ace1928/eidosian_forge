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
@classmethod
def _calculate_sha_(cls, repo: 'Repo', commit: 'Commit') -> bytes:
    """Calculate the sha of a commit.

        :param repo: Repo object the commit should be part of
        :param commit: Commit object for which to generate the sha
        """
    stream = BytesIO()
    commit._serialize(stream)
    streamlen = stream.tell()
    stream.seek(0)
    istream = repo.odb.store(IStream(cls.type, streamlen, stream))
    return istream.binsha