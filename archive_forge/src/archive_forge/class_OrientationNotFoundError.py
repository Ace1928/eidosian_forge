from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import exit_code
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
class OrientationNotFoundError(TestingError):
    """Failed to find an orientation in the test environment catalog."""

    def __init__(self, orientation):
        super(OrientationNotFoundError, self).__init__("'{o}' is not a valid device orientation".format(o=orientation))