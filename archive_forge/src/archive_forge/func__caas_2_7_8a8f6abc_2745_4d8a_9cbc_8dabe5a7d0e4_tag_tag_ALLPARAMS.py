import sys
import unittest
from types import GeneratorType
import pytest
from libcloud.test import MockHttp
from libcloud.utils.py3 import ET, httplib
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeLocation, NodeAuthPassword
from libcloud.test.secrets import NTTCIS_PARAMS
from libcloud.common.nttcis import (
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.nttcis import NttCisNic
from libcloud.compute.drivers.nttcis import NttCisNodeDriver as NttCis
def _caas_2_7_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_tag_tag_ALLPARAMS(self, method, url, body, headers):
    _, params = url.split('?')
    parameters = params.split('&')
    for parameter in parameters:
        key, value = parameter.split('=')
        if key == 'assetId':
            assert value == 'fake_asset_id'
        elif key == 'assetType':
            assert value == 'fake_asset_type'
        elif key == 'valueRequired':
            assert value == 'false'
        elif key == 'displayOnReport':
            assert value == 'false'
        elif key == 'pageSize':
            assert value == '250'
        elif key == 'datacenterId':
            assert value == 'fake_location'
        elif key == 'value':
            assert value == 'fake_value'
        elif key == 'tagKeyName':
            assert value == 'fake_tag_key_name'
        elif key == 'tagKeyId':
            assert value == 'fake_tag_key_id'
        else:
            raise ValueError('Could not find in url parameters {}:{}'.format(key, value))
    body = self.fixtures.load('tag_tag_list.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])