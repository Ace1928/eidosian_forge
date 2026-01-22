from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import logging
from googlecloudsdk.core import exceptions
def StartNewStatement(self):
    self.ddl_parts_ = []
    self.statements_.append(self.ddl_parts_)