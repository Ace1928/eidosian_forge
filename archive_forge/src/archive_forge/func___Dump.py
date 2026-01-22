from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.core.resource import resource_printer_base
import six
def __Dump(self, resource):
    data = json.dumps(resource, ensure_ascii=False, indent=resource_printer_base.STRUCTURED_INDENTATION, separators=(',', ': '), sort_keys=True)
    return six.text_type(data)