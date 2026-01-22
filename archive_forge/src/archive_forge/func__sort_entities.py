from gitdb.db.base import (
from gitdb.util import LazyMixin
from gitdb.exc import (
from gitdb.pack import PackEntity
from functools import reduce
import os
import glob
def _sort_entities(self):
    self._entities.sort(key=lambda l: l[0], reverse=True)