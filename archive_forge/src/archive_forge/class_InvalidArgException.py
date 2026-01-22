from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import exit_code
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
class InvalidArgException(calliope_exceptions.InvalidArgumentException):
    """InvalidArgException is for malformed gcloud firebase test argument values.

  It provides a wrapper around Calliope's InvalidArgumentException that
  conveniently converts internal arg names with underscores into the external
  arg names with hyphens.
  """

    def __init__(self, param_name, message):
        super(InvalidArgException, self).__init__(ExternalArgNameFrom(param_name), message)