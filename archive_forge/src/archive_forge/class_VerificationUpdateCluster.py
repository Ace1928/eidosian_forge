import sys
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_log import log as logging
from saharaclient.osc import utils
from saharaclient.osc.v1 import clusters as c_v1
class VerificationUpdateCluster(c_v1.VerificationUpdateCluster):
    """Updates cluster verifications"""
    log = logging.getLogger(__name__ + '.VerificationUpdateCluster')