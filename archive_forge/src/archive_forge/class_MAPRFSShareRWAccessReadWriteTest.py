import ast
import ddt
from tempest.lib import exceptions as tempest_lib_exc
from manilaclient import api_versions
from manilaclient import config
from manilaclient.tests.functional import base
class MAPRFSShareRWAccessReadWriteTest(ShareAccessReadWriteBase):
    protocol = 'maprfs'
    access_level = 'rw'