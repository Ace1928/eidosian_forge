import json
from unittest import mock
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
class QueuesV2HealthFunctionalTest(base.QueuesTestBase):

    def test_ping(self):
        self.assertTrue(self.client.ping())

    def test_health(self):
        health = self.client.health()
        self.assertIsInstance(health, dict)