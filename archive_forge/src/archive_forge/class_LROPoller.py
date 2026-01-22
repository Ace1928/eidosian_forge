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
class LROPoller(waiter.OperationPoller):
    """Polls on completion of an Apigee long running operation."""

    def __init__(self, organization):
        super(LROPoller, self).__init__()
        self.organization = organization

    def IsDone(self, operation):
        finished = False
        try:
            finished = operation['metadata']['state'] == 'FINISHED'
        except KeyError as err:
            raise waiter.OperationError('Malformed operation; %s\n%r' % (err, operation))
        if finished and 'error' in operation:
            raise errors.RequestError('operation', {'name': operation['name']}, 'await', body=json.dumps(operation))
        return finished

    def Poll(self, operation_uuid):
        return OperationsClient.Describe({'organizationsId': self.organization, 'operationsId': operation_uuid})

    def GetResult(self, operation):
        if 'response' in operation:
            return operation['response']
        return None