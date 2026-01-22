from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.core.resource import custom_printer_base as cp
from googlecloudsdk.core.resource import yaml_printer as yp
def _ClearField(self, registration, field):
    if field in self._KNOWN_REPEATED_FIELDS:
        setattr(registration, field, [])
    else:
        setattr(registration, field, None)