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
def trailers(self) -> Dict[str, str]:
    """Get the trailers of the message as a dictionary

        :note: This property is deprecated, please use either :attr:`trailers_list` or
            :attr:`trailers_dict``.

        :return:
            Dictionary containing whitespace stripped trailer information. Only contains
            the latest instance of each trailer key.
        """
    return {k: v[0] for k, v in self.trailers_dict.items()}