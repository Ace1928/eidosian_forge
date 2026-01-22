from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def IsNotDone(self, restore, unused_state):
    del unused_state
    return not (restore.state == self.messages.Restore.StateValueValuesEnum.SUCCEEDED or restore.state == self.messages.Restore.StateValueValuesEnum.FAILED or restore.state == self.messages.Restore.StateValueValuesEnum.DELETING)