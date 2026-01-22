from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import re
from googlecloudsdk.api_lib.apigee import base
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.apigee import errors
from googlecloudsdk.command_lib.apigee import request
from googlecloudsdk.command_lib.apigee import resource_args
from googlecloudsdk.core import log
class DeploymentsClient(object):

    @classmethod
    def List(cls, identifiers):
        """Returns a list of deployments, filtered by `identifiers`.

    The deployment-listing API, unlike most GCP APIs, is very flexible as to
    what kinds of objects are provided as the deployments' parents. An
    organization is required, but any combination of environment, proxy or
    shared flow, and API revision can be given in addition to that.

    Args:
      identifiers: dictionary with fields that describe which deployments to
        list. `organizationsId` is required. `environmentsId`, `apisId`, and
        `revisionsId` can be optionally provided to further filter the list.
        Shared flows are not yet supported.

    Returns:
      A list of Apigee deployments, each represented by a parsed JSON object.
    """
        identifier_names = ['organization', 'environment', 'api', 'revision']
        entities = [resource_args.ENTITIES[name] for name in identifier_names]
        entity_path = []
        for entity in entities:
            key = entity.plural + 'Id'
            if key in identifiers and identifiers[key] is not None:
                entity_path.append(entity.singular)
        if 'revision' in entity_path and 'api' not in entity_path:
            entity_path.remove('revision')
        try:
            response = request.ResponseToApiRequest(identifiers, entity_path, 'deployment')
        except errors.EntityNotFoundError:
            response = []
        if 'apiProxy' in response:
            return [response]
        if 'deployments' in response:
            return response['deployments']
        if not response:
            return []
        return response