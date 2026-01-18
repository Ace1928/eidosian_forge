from __future__ import annotations
import importlib
import json
import pathlib
import platform
import sys
import click
import pytest
import yaml
from click.testing import CliRunner
import dask
import dask.cli
from dask._compatibility import importlib_metadata
def test_repeated_name_registration_warn():
    from dask.cli import _register_command_ep
    one = importlib_metadata.EntryPoint(name='one', value='dask.tests.test_cli:good_command', group='dask_cli')
    two = importlib_metadata.EntryPoint(name='two', value='dask.tests.test_cli:good_command_2', group='dask_cli')
    _register_command_ep(dummy_cli_2, one)
    with pytest.warns(UserWarning, match='While registering the command with name'):
        _register_command_ep(dummy_cli_2, two)