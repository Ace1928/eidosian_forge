from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import logging
from googlecloudsdk.core import exceptions
def EmitBuffer(self):
    if self.mode_start_index_ >= self.next_index_:
        return
    self.ddl_parts_.append(self.ddl_[self.mode_start_index_:self.next_index_])
    self.SkipBuffer()
    self.logger_.debug('emitted: %s', self.ddl_parts_[-1])