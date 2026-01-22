from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.core.console import progress_tracker
class CancellationPoller(waiter.OperationPoller):
    """Polls for cancellation of a resource."""

    def __init__(self, getter):
        """Supply getter as the resource getter."""
        self._getter = getter
        self._ret = None

    def IsDone(self, obj):
        return obj is None or obj.conditions.IsTerminal()

    def Poll(self, ref):
        self._ret = self._getter(ref)
        return self._ret

    def GetMessage(self):
        if self._ret and self._ret.conditions:
            return self._ret.conditions.DescriptiveMessage() or ''
        return ''

    def GetResult(self, obj):
        return obj