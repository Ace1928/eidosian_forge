import configparser
import os
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import ironicclient.tests.functional.utils as utils
def assertNodeValidate(self, node_validate):
    """Assert that all interfaces present are valid.

        :param node_validate: output from node-validate cmd
        """
    self.assertNotIn('False', [x['Result'] for x in node_validate])