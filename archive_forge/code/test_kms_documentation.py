from boto.compat import json
from boto.kms.layer1 import KMSConnection
from tests.unit import AWSMockServiceTestCase

        This test ensures that the output is base64 decoded before
        it is returned to the user.
        