from unittest import mock
from neutronclient.common import exceptions
from neutronclient.v2_0 import client as neutronclient
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class L2GatewaySegmentationRequired(exceptions.NeutronException):
    message = 'L2 gateway segmentation id must be consistent for all the interfaces'