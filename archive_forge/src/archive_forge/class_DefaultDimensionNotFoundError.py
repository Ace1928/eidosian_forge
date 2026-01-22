from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import exit_code
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
class DefaultDimensionNotFoundError(TestingError):
    """Failed to find a 'default' tag on any value for a device dimension."""

    def __init__(self, dim_name):
        super(DefaultDimensionNotFoundError, self).__init__("Test Lab did not provide a default value for '{d}'".format(d=dim_name))