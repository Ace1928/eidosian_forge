from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class MissingRequiredArgumentError(Error):
    """Error when a required concept can't be found."""

    def __init__(self, concept_name, message):
        super(MissingRequiredArgumentError, self).__init__('No value was provided for [{}]: {}'.format(concept_name, message))