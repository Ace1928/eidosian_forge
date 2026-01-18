import random
import time
from functools import partial
from tests.compat import unittest
from boto.beanstalk.wrapper import Layer1Wrapper
import boto.beanstalk.response as response
def test_restart_app_server(self):
    result = self.beanstalk.restart_app_server(environment_name=self.environment)
    self.assertIsInstance(result, response.RestartAppServerResponse)
    self.wait_for_env(self.environment)