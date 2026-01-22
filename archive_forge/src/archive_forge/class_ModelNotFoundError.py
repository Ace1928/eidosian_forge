from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import exit_code
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
class ModelNotFoundError(TestingError):
    """Failed to find a device model in the test environment catalog."""

    def __init__(self, model_id):
        super(ModelNotFoundError, self).__init__("'{id}' is not a valid model".format(id=model_id))