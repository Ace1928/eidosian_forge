from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class TPUInUnusableTerminalState(exceptions.Error):
    """Error when the TPU is in an unusable, terminal state."""

    def __init__(self, state):
        super(TPUInUnusableTerminalState, self).__init__('This TPU has terminal state "{}", so it cannot be used anymore.'.format(state))