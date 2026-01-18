import os
import time
import pytest
import srsly
from weasel.cli.remote_storage import RemoteStorage
from weasel.schemas import ProjectConfigSchema, validate
from weasel.util import is_subpath_of, load_project_config, make_tempdir
from weasel.util import validate_project_commands
def test_project_config_validation_full():
    config = {'vars': {'some_var': 20}, 'directories': ['assets', 'configs', 'corpus', 'scripts', 'training'], 'assets': [{'dest': 'x', 'extra': True, 'url': 'https://example.com', 'checksum': '63373dd656daa1fd3043ce166a59474c'}, {'dest': 'y', 'git': {'repo': 'https://github.com/example/repo', 'branch': 'develop', 'path': 'y'}}, {'dest': 'z', 'extra': False, 'url': 'https://example.com', 'checksum': '63373dd656daa1fd3043ce166a59474c'}], 'commands': [{'name': 'train', 'help': 'Train a model', 'script': ['python -m spacy train config.cfg -o training'], 'deps': ['config.cfg', 'corpus/training.spcy'], 'outputs': ['training/model-best']}, {'name': 'test', 'script': ['pytest', 'custom.py'], 'no_skip': True}], 'workflows': {'all': ['train', 'test'], 'train': ['train']}}
    errors = validate(ProjectConfigSchema, config)
    assert not errors