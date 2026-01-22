from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import exit_code
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
class NetworkProfileNotFoundError(TestingError):
    """Failed to find a network profile in the test environment catalog."""

    def __init__(self, profile_id):
        super(NetworkProfileNotFoundError, self).__init__("Could not find network profile ID '{id}'".format(id=profile_id))