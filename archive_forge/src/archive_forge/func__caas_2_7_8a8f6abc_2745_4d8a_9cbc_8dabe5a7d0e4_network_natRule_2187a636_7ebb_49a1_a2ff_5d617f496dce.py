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
def _caas_2_7_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_network_natRule_2187a636_7ebb_49a1_a2ff_5d617f496dce(self, method, url, body, headers):
    body = self.fixtures.load('network_natRule_2187a636_7ebb_49a1_a2ff_5d617f496dce.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])