from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
def _ValidateAllFieldsRecognized(path, conditions):
    unrecognized_fields = set()
    for condition in conditions:
        if condition.all_unrecognized_fields():
            unrecognized_fields.update(condition.all_unrecognized_fields())
    if unrecognized_fields:
        raise InvalidFormatError(path, 'Unrecognized fields: [{}]'.format(', '.join(unrecognized_fields)), type(conditions[0]))