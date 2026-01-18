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
@mock.patch('kubernetes.config.kube_config.OAuth2Session.refresh_token')
@mock.patch('kubernetes.config.kube_config.ApiClient.request')
def test_oidc_with_refresh_nocert(self, mock_ApiClient, mock_OAuth2Session):
    mock_response = mock.MagicMock()
    type(mock_response).status = mock.PropertyMock(return_value=200)
    type(mock_response).data = mock.PropertyMock(return_value=json.dumps({'token_endpoint': 'https://example.org/identity/token'}))
    mock_ApiClient.return_value = mock_response
    mock_OAuth2Session.return_value = {'id_token': 'abc123', 'refresh_token': 'newtoken123'}
    loader = KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='expired_oidc_nocert')
    self.assertTrue(loader._load_auth_provider_token())
    self.assertEqual('Bearer abc123', loader.token)