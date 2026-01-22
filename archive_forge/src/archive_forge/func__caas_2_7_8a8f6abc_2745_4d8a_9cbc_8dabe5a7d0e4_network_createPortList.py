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
def _caas_2_7_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_network_createPortList(self, method, url, body, headers):
    request = ET.fromstring(body)
    if request.tag != '{urn:didata.com:api:cloud:types}createPortList':
        raise InvalidRequestError(request.tag)
    net_domain = findtext(request, 'networkDomainId', TYPES_URN)
    if net_domain is None:
        raise ValueError('Network Domain should not be empty')
    ports_required = findall(request, 'port', TYPES_URN)
    child_port_list_required = findall(request, 'childPortListId', TYPES_URN)
    if 0 == len(ports_required) and 0 == len(child_port_list_required):
        raise ValueError('At least one port element or one childPortListId element must be provided')
    if ports_required[0].get('begin') is None:
        raise ValueError('PORT begin value should not be empty')
    body = self.fixtures.load('port_list_create.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])