from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Any, Dict
def additional_properties_to_dict(spec):
    """Extracts the additional properties of a message.Message as a dictionary."""
    if spec is None:
        return {}
    return {prop.key: prop.value for prop in spec.additionalProperties}