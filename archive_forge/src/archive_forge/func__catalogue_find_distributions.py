import os
import re
import abc
import csv
import sys
import zipp
import email
import pathlib
import operator
import functools
import itertools
import posixpath
import collections
from ._compat import (
from configparser import ConfigParser
from contextlib import suppress
from importlib import import_module
from importlib.abc import MetaPathFinder
from itertools import starmap
from typing import Any, List, Mapping, TypeVar, Union
def _catalogue_find_distributions(self, context=DistributionFinder.Context()):
    """
        Find distributions.

        Return an iterable of all Distribution instances capable of
        loading the metadata for packages matching ``context.name``
        (or all names if ``None`` indicated) along the paths in the list
        of directories ``context.path``.
        """
    found = self._search_paths(context.name, context.path)
    return map(PathDistribution, found)