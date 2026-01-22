from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.core import exceptions
class RevisionNameBasedPoller(waiter.OperationPoller):
    """Poll for the revision with the given name to exist."""

    def __init__(self, operations, revision_ref_getter):
        self._operations = operations
        self._revision_ref_getter = revision_ref_getter

    def IsDone(self, revision_obj):
        return bool(revision_obj)

    def Poll(self, revision_name):
        revision_ref = self._revision_ref_getter(revision_name)
        return self._operations.GetRevision(revision_ref)

    def GetResult(self, revision_obj):
        return revision_obj