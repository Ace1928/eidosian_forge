from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.command_lib.apigee import errors
from googlecloudsdk.command_lib.apigee import resource_args
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import requests
from six.moves import urllib
def _ResourceIdentifier(identifiers, entity_path):
    """Returns an OrderedDict uniquely identifying the resource to be accessed.

  Args:
    identifiers: a collection that maps entity type names to identifiers.
    entity_path: a list of entity type names from least to most specific.

  Raises:
    MissingIdentifierError: an entry in entity_path is missing from
      `identifiers`.
  """
    resource_identifier = collections.OrderedDict()
    for entity_name in entity_path:
        entity = resource_args.ENTITIES[entity_name]
        id_key = entity.plural + 'Id'
        if id_key not in identifiers or identifiers[id_key] is None:
            raise errors.MissingIdentifierError(entity.singular)
        resource_identifier[entity] = identifiers[id_key]
    return resource_identifier