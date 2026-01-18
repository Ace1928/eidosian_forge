import os
import requests
import subprocess
import time
import uuid
import concurrent.futures
from oslo_config import cfg
from testtools import matchers
import oslo_messaging
from oslo_messaging.tests.functional import utils
def test_specific_server(self):
    group = self.useFixture(utils.RpcServerGroupFixture(self.conf, self.rpc_url))
    client = group.client(1, cast=True)
    client.append(text='open')
    client.append(text='stack')
    client.add(increment=2)
    client.add(increment=10)
    time.sleep(0.3)
    client.sync()
    group.sync(1)
    self.assertIn(group.servers[1].endpoint.sval, ['openstack', 'stackopen'])
    self.assertEqual(12, group.servers[1].endpoint.ival)
    for i in [0, 2]:
        self.assertEqual('', group.servers[i].endpoint.sval)
        self.assertEqual(0, group.servers[i].endpoint.ival)