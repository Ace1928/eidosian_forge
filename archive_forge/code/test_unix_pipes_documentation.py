from __future__ import annotations
import errno
import os
import select
import sys
from typing import TYPE_CHECKING
import pytest
from .. import _core
from .._core._tests.tutil import gc_collect_harder, skip_if_fbsd_pipes_broken
from ..testing import check_one_way_stream, wait_all_tasks_blocked
Makes a new pair of pipes.