from __future__ import absolute_import, division, print_function
import json
import sys
from awx.main.models import Organization, Team, Project, Inventory
from requests.models import Response
from unittest import mock
def test_version_warning_strictness_controller(collection_import, silence_warning):
    ControllerAPIModule = collection_import('plugins.module_utils.controller_api').ControllerAPIModule
    cli_data = {'ANSIBLE_MODULE_ARGS': {}}
    testargs = ['module_file2.py', json.dumps(cli_data)]
    with mock.patch.object(sys, 'argv', testargs):
        with mock.patch('ansible.module_utils.urls.Request.open', new=mock_controller_ping_response):
            my_module = ControllerAPIModule(argument_spec=dict())
            my_module._COLLECTION_VERSION = '1.2.0'
            my_module._COLLECTION_TYPE = 'controller'
            my_module.get_endpoint('ping')
    silence_warning.assert_not_called()
    with mock.patch.object(sys, 'argv', testargs):
        with mock.patch('ansible.module_utils.urls.Request.open', new=mock_controller_ping_response):
            my_module = ControllerAPIModule(argument_spec=dict())
            my_module._COLLECTION_VERSION = '1.0.0'
            my_module._COLLECTION_TYPE = 'controller'
            my_module.get_endpoint('ping')
    silence_warning.assert_called_once_with('You are running collection version {0} but connecting to {1} version {2}'.format(my_module._COLLECTION_VERSION, controller_name, ping_version))