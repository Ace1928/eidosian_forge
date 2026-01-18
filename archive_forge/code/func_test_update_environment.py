import random
import time
from functools import partial
from tests.compat import unittest
from boto.beanstalk.wrapper import Layer1Wrapper
import boto.beanstalk.response as response
def test_update_environment(self):
    result = self.beanstalk.update_environment(environment_name=self.environment)
    self.assertIsInstance(result, response.UpdateEnvironmentResponse)
    self.wait_for_env(self.environment)