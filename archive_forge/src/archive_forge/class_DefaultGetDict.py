import logging  # noqa
from collections import defaultdict
import io
import os
import re
import shlex
import sys
import traceback
import distutils.ccompiler
from distutils import errors
from distutils import log
import pkg_resources
from setuptools import dist as st_dist
from setuptools import extension
from pbr import extra_files
import pbr.hooks
class DefaultGetDict(defaultdict):
    """Like defaultdict, but get() also sets and returns the default value."""

    def get(self, key, default=None):
        if default is None:
            default = self.default_factory()
        return super(DefaultGetDict, self).setdefault(key, default)