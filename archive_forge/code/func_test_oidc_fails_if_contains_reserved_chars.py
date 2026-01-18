import base64
import datetime
import json
import os
import shutil
import tempfile
import unittest
import mock
from ruamel import yaml
from six import PY3, next
from kubernetes.client import Configuration
from .config_exception import ConfigException
from .kube_config import (ENV_KUBECONFIG_PATH_SEPARATOR, ConfigNode, FileOrData,
def test_oidc_fails_if_contains_reserved_chars(self):
    loader = KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='oidc_contains_reserved_character')
    self.assertEqual(loader._load_oid_token('oidc_contains_reserved_character'), None)