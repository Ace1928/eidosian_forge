from __future__ import annotations
import collections
import copy
import datetime
import os
import sys
import typing as t
import nbformat
from nbformat import NotebookNode, validator
from traitlets import Bool, HasTraits, List, TraitError, Unicode
from traitlets.config import Config
from traitlets.config.configurable import LoggingConfigurable
from traitlets.utils.importstring import import_item
class ResourcesDict(collections.defaultdict):
    """A default dict for resources."""

    def __missing__(self, key):
        """Handle missing value."""
        return ''