import configparser
import os
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import ironicclient.tests.functional.utils as utils
def assertNodeStates(self, node_show, node_show_states):
    """Assert that node_show_states output corresponds to node_show output.

        :param node_show: output from node-show cmd
        :param node_show_states: output from node-show-states cmd
        """
    for key in node_show_states.keys():
        self.assertEqual(node_show_states[key], node_show[key])