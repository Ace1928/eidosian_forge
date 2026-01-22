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

        Search the commit message for any co-authors of this commit.

        Details on co-authors: https://github.blog/2018-01-29-commit-together-with-co-authors/

        :return: List of co-authors for this commit (as Actor objects).
        