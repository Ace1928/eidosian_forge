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
def test_start_drs_failover_invalid_status(driver):
    NttCisMockHttp.type = 'INVALID_STATUS'
    cg_id = '195a426b-4559-4c79-849e-f22cdf2bfb6e'
    with pytest.raises(NttCisAPIException) as excinfo:
        driver.ex_initiate_drs_failover(cg_id)
    assert 'INVALID_STATUS' in excinfo.value.code