from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import exit_code
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
class VersionNotFoundError(TestingError):
    """Failed to find an OS version in the test environment catalog."""

    def __init__(self, version):
        super(VersionNotFoundError, self).__init__("'{v}' is not a valid OS version".format(v=version))