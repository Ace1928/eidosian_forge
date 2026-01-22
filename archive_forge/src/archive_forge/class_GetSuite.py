import random
import time
from functools import partial
from tests.compat import unittest
from boto.beanstalk.wrapper import Layer1Wrapper
import boto.beanstalk.response as response
class GetSuite(BasicSuite):

    def test_describe_applications(self):
        result = self.beanstalk.describe_applications()
        self.assertIsInstance(result, response.DescribeApplicationsResponse)

    def test_describe_application_versions(self):
        result = self.beanstalk.describe_application_versions()
        self.assertIsInstance(result, response.DescribeApplicationVersionsResponse)

    def test_describe_configuration_options(self):
        result = self.beanstalk.describe_configuration_options()
        self.assertIsInstance(result, response.DescribeConfigurationOptionsResponse)

    def test_12_describe_environments(self):
        result = self.beanstalk.describe_environments()
        self.assertIsInstance(result, response.DescribeEnvironmentsResponse)

    def test_14_describe_events(self):
        result = self.beanstalk.describe_events()
        self.assertIsInstance(result, response.DescribeEventsResponse)

    def test_15_list_available_solution_stacks(self):
        result = self.beanstalk.list_available_solution_stacks()
        self.assertIsInstance(result, response.ListAvailableSolutionStacksResponse)
        self.assertIn('32bit Amazon Linux running Tomcat 6', result.solution_stacks)