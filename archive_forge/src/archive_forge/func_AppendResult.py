from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.resource import resource_transform
import six
def AppendResult(name, result):
    """Appends key/value pairs from obj into res.

    Adds projection label if defined.

    Args:
      name: Name of key.
      result: Value of key in obj.
    """
    use_legacy = properties.VALUES.core.use_legacy_flattened_format.GetBool()
    if not use_legacy and labels and (name in labels):
        res.append((labels[name].lower(), result))
    else:
        res.append((name, result))