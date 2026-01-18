from mmap import mmap
import re
import time as _time
from git.compat import defenc
from git.objects.util import (
from git.util import (
import os.path as osp
from typing import Iterator, List, Tuple, Union, TYPE_CHECKING
from git.types import PathLike
Write this instance's data to the file we are originating from.

        :return: self
        