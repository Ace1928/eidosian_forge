import unittest
from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.client import DesignateCLI
from designateclient.functionaltests.datagen import random_zone_name
from designateclient.functionaltests.v2.fixtures import TransferRequestFixture
from designateclient.functionaltests.v2.fixtures import ZoneFixture
def test_zone_transfer_accept_list(self):
    self.useFixture(TransferRequestFixture(self.zone))
    list_transfer_accepts = self.clients.zone_transfer_accept_list()
    self.assertGreater(len(list_transfer_accepts), 0)