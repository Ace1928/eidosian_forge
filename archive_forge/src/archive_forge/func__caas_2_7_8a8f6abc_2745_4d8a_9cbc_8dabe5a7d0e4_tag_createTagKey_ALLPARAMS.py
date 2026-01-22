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
def _caas_2_7_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_tag_createTagKey_ALLPARAMS(self, method, url, body, headers):
    request = ET.fromstring(body)
    if request.tag != '{urn:didata.com:api:cloud:types}createTagKey':
        raise InvalidRequestError(request.tag)
    name = findtext(request, 'name', TYPES_URN)
    description = findtext(request, 'description', TYPES_URN)
    value_required = findtext(request, 'valueRequired', TYPES_URN)
    display_on_report = findtext(request, 'displayOnReport', TYPES_URN)
    if name is None:
        raise ValueError('Name must have a value in the request')
    if description is None:
        raise ValueError('Description should have a value')
    if value_required is None or value_required != 'false':
        raise ValueError('valueRequired should be false')
    if display_on_report is None or display_on_report != 'false':
        raise ValueError('displayOnReport should be false')
    body = self.fixtures.load('tag_createTagKey.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])