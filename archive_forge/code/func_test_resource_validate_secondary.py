from unittest import mock
from heat.common import exception
from heat.engine.resources.openstack.designate import zone
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_resource_validate_secondary(self):
    self._test_resource_validate(zone.DesignateZone.SECONDARY, zone.DesignateZone.MASTERS)