from __future__ import annotations
import sys
from importlib.metadata import EntryPoint
from unittest import mock
import pytest
from xarray.backends import common, plugins
from xarray.tests import (
importlib_metadata_mock = "importlib.metadata"
def test_set_missing_parameters_raise_error() -> None:
    backend = DummyBackendEntrypointKwargs
    with pytest.raises(TypeError):
        plugins.set_missing_parameters({'engine': backend})
    backend_args = DummyBackendEntrypointArgs
    with pytest.raises(TypeError):
        plugins.set_missing_parameters({'engine': backend_args})