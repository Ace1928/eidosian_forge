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
def _caas_2_7_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_server_server_ALLFILTERS(self, method, url, body, headers):
    _, params = url.split('?')
    parameters = params.split('&')
    for parameter in parameters:
        key, value = parameter.split('=')
        if key == 'datacenterId':
            assert value == 'fake_loc'
        elif key == 'networkId':
            assert value == 'fake_network'
        elif key == 'networkDomainId':
            assert value == 'fake_network_domain'
        elif key == 'vlanId':
            assert value == 'fake_vlan'
        elif key == 'ipv6':
            assert value == 'fake_ipv6'
        elif key == 'privateIpv4':
            assert value == 'fake_ipv4'
        elif key == 'name':
            assert value == 'fake_name'
        elif key == 'state':
            assert value == 'fake_state'
        elif key == 'started':
            assert value == 'True'
        elif key == 'deployed':
            assert value == 'True'
        elif key == 'sourceImageId':
            assert value == 'fake_image'
        else:
            raise ValueError('Could not find in url parameters {}:{}'.format(key, value))
    body = self.fixtures.load('server_server.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])