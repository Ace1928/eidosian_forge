from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.core import exceptions
import six
class RepeatedConceptName(Error):
    """Raised when adding a concept if one with the given name already exists."""

    def __init__(self, concept_name):
        msg = 'Repeated concept name [{}].'.format(concept_name)
        super(RepeatedConceptName, self).__init__(msg)