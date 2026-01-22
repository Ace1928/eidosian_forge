from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import six.moves.urllib.parse
def _validation_exception(self, name):
    return BadNameException('Docker image name must be fully qualified (e.g.registry/repository@digest) saw: %s' % name)