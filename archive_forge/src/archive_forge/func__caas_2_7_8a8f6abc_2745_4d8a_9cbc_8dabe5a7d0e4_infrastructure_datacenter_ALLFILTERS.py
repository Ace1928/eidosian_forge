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
def _caas_2_7_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_infrastructure_datacenter_ALLFILTERS(self, method, url, body, headers):
    if url.endswith('id=NA9'):
        body = self.fixtures.load('infrastructure_datacenter_NA9.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])
    body = self.fixtures.load('infrastructure_datacenter.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])