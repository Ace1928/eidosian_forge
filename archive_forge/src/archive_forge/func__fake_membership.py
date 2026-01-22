from unittest import mock
from oslo_config import cfg
from glance import context
from glance.tests.unit import utils as unit_utils
from glance.tests import utils
def _fake_membership(can_share=False):
    return {'can_share': can_share}