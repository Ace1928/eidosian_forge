from tests.integration.route53 import Route53TestCase
from boto.compat import six
from boto.route53.healthcheck import HealthCheck
from boto.route53.record import ResourceRecordSets

        Test that health checks cannot be created with an invalid
        'failure_threshold'.
        