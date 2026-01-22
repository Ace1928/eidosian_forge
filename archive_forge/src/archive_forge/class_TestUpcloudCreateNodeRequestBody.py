import sys
import json
from unittest.mock import Mock, call
from libcloud.test import unittest
from libcloud.compute.base import NodeSize, NodeImage, NodeLocation, NodeAuthSSHKey
from libcloud.common.upcloud import (
class TestUpcloudCreateNodeRequestBody(unittest.TestCase):

    def setUp(self):
        self.image = NodeImage(id='01000000-0000-4000-8000-000030060200', name='Ubuntu Server 16.04 LTS (Xenial Xerus)', driver='', extra={'type': 'template'})
        self.location = NodeLocation(id='fi-hel1', name='Helsinki #1', country='FI', driver='')
        self.size = NodeSize(id='1xCPU-1GB', name='1xCPU-1GB', ram=1024, disk=30, bandwidth=2048, extra={'core_number': 1, 'storage_tier': 'maxiops'}, price=None, driver='')

    def test_creating_node_from_template_image(self):
        body = UpcloudCreateNodeRequestBody(name='ts', image=self.image, location=self.location, size=self.size)
        json_body = body.to_json()
        dict_body = json.loads(json_body)
        expected_body = {'server': {'title': 'ts', 'hostname': 'localhost', 'plan': '1xCPU-1GB', 'zone': 'fi-hel1', 'login_user': {'username': 'root', 'create_password': 'yes'}, 'storage_devices': {'storage_device': [{'action': 'clone', 'title': 'Ubuntu Server 16.04 LTS (Xenial Xerus)', 'storage': '01000000-0000-4000-8000-000030060200', 'size': 30, 'tier': 'maxiops'}]}}}
        self.assertDictEqual(expected_body, dict_body)

    def test_creating_node_from_cdrom_image(self):
        image = NodeImage(id='01000000-0000-4000-8000-000030060200', name='Ubuntu Server 16.04 LTS (Xenial Xerus)', driver='', extra={'type': 'cdrom'})
        body = UpcloudCreateNodeRequestBody(name='ts', image=image, location=self.location, size=self.size)
        json_body = body.to_json()
        dict_body = json.loads(json_body)
        expected_body = {'server': {'title': 'ts', 'hostname': 'localhost', 'plan': '1xCPU-1GB', 'zone': 'fi-hel1', 'login_user': {'username': 'root', 'create_password': 'yes'}, 'storage_devices': {'storage_device': [{'action': 'create', 'size': 30, 'tier': 'maxiops', 'title': 'Ubuntu Server 16.04 LTS (Xenial Xerus)'}, {'action': 'attach', 'storage': '01000000-0000-4000-8000-000030060200', 'type': 'cdrom'}]}}}
        self.assertDictEqual(expected_body, dict_body)

    def test_creating_node_using_ssh_keys(self):
        auth = NodeAuthSSHKey('sshkey')
        body = UpcloudCreateNodeRequestBody(name='ts', image=self.image, location=self.location, size=self.size, auth=auth)
        json_body = body.to_json()
        dict_body = json.loads(json_body)
        expected_body = {'server': {'title': 'ts', 'hostname': 'localhost', 'plan': '1xCPU-1GB', 'zone': 'fi-hel1', 'login_user': {'username': 'root', 'ssh_keys': {'ssh_key': ['sshkey']}}, 'storage_devices': {'storage_device': [{'action': 'clone', 'size': 30, 'title': 'Ubuntu Server 16.04 LTS (Xenial Xerus)', 'tier': 'maxiops', 'storage': '01000000-0000-4000-8000-000030060200'}]}}}
        self.assertDictEqual(expected_body, dict_body)

    def test_creating_node_using_hostname(self):
        body = UpcloudCreateNodeRequestBody(name='ts', image=self.image, location=self.location, size=self.size, ex_hostname='myhost.upcloud.com')
        json_body = body.to_json()
        dict_body = json.loads(json_body)
        expected_body = {'server': {'title': 'ts', 'hostname': 'myhost.upcloud.com', 'plan': '1xCPU-1GB', 'zone': 'fi-hel1', 'login_user': {'username': 'root', 'create_password': 'yes'}, 'storage_devices': {'storage_device': [{'action': 'clone', 'title': 'Ubuntu Server 16.04 LTS (Xenial Xerus)', 'storage': '01000000-0000-4000-8000-000030060200', 'tier': 'maxiops', 'size': 30}]}}}
        self.assertDictEqual(expected_body, dict_body)

    def test_creating_node_with_non_default_username(self):
        body = UpcloudCreateNodeRequestBody(name='ts', image=self.image, location=self.location, size=self.size, ex_username='someone')
        json_body = body.to_json()
        dict_body = json.loads(json_body)
        login_user = dict_body['server']['login_user']
        self.assertDictEqual({'username': 'someone', 'create_password': 'yes'}, login_user)