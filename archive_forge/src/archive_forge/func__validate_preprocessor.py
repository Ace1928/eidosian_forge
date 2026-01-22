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
def _validate_preprocessor(self, nbc, preprocessor):
    try:
        nbformat.validate(nbc, relax_add_props=True)
    except nbformat.ValidationError:
        self.log.error('Notebook is invalid after preprocessor %s', preprocessor)
        raise