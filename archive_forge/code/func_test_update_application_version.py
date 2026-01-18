import random
import time
from functools import partial
from tests.compat import unittest
from boto.beanstalk.wrapper import Layer1Wrapper
import boto.beanstalk.response as response
def test_update_application_version(self):
    self.create_application()
    self.beanstalk.create_application_version(application_name=self.app_name, version_label=self.app_version)
    result = self.beanstalk.update_application_version(application_name=self.app_name, version_label=self.app_version)
    self.assertIsInstance(result, response.UpdateApplicationVersionResponse)