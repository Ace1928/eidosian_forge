from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def MakePreservedStateMetadataMapEntry(messages, key, value):
    """Make a map entry for metadata field in preservedState message."""
    return messages.PreservedState.MetadataValue.AdditionalProperty(key=key, value=value)