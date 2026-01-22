from __future__ import annotations
import sys
from importlib.metadata import EntryPoint
from unittest import mock
import pytest
from xarray.backends import common, plugins
from xarray.tests import (
importlib_metadata_mock = "importlib.metadata"
class DummyBackendEntrypoint1(common.BackendEntrypoint):

    def open_dataset(self, filename_or_obj, *, decoder):
        pass