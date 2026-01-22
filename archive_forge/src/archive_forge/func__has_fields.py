import argparse
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
@staticmethod
def _has_fields(rule, required_fields):
    return all([key in rule for key in required_fields])