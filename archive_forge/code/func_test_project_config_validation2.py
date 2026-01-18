import os
import time
import pytest
import srsly
from weasel.cli.remote_storage import RemoteStorage
from weasel.schemas import ProjectConfigSchema, validate
from weasel.util import is_subpath_of, load_project_config, make_tempdir
from weasel.util import validate_project_commands
@pytest.mark.parametrize('config,n_errors', [({'commands': {'a': []}}, 1), ({'commands': [{'help': '...'}]}, 1), ({'commands': [{'name': 'a', 'extra': 'b'}]}, 1), ({'commands': [{'extra': 'b'}]}, 2), ({'commands': [{'name': 'a', 'deps': [123]}]}, 1)])
def test_project_config_validation2(config, n_errors):
    errors = validate(ProjectConfigSchema, config)
    assert len(errors) == n_errors