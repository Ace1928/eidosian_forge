import json
import tempfile
from unittest import mock
import io
from oslo_serialization import base64
import testtools
from testtools import matchers
from urllib import error
import yaml
from heatclient.common import template_utils
from heatclient.common import utils
from heatclient import exc
def test_process_multiple_environment_files_container(self):
    env_list_tracker = []
    env_paths = ['/home/my/dir/env.yaml']
    files, env = template_utils.process_multiple_environments_and_files(env_paths, env_list_tracker=env_list_tracker, fetch_env_files=False)
    self.assertEqual(env_paths, env_list_tracker)
    self.assertEqual({}, files)
    self.assertEqual({}, env)