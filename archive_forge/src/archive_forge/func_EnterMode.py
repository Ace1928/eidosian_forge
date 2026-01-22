from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import logging
from googlecloudsdk.core import exceptions
def EnterMode(self, mode):
    self.logger_.debug('enter mode: %s at index: %d', mode.enter_seq_, self.next_index_)
    self.mode_ = mode