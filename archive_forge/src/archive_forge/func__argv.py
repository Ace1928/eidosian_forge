import os
import operator
import sys
import contextlib
import itertools
import unittest
from distutils.errors import DistutilsError, DistutilsOptionError
from distutils import log
from unittest import TestLoader
from pkg_resources import (
from .._importlib import metadata
from setuptools import Command
from setuptools.extern.more_itertools import unique_everseen
from setuptools.extern.jaraco.functools import pass_none
@property
def _argv(self):
    return ['unittest'] + self.test_args