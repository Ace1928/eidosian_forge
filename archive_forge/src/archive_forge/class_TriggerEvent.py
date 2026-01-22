from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
import enum
class TriggerEvent(object):
    """Represents --trigger-event flag value options."""
    OPTIONAL_RESOURCE_TYPES = [Resources.PROJECT]

    def __init__(self, label, resource_type):
        self.label = label
        self.resource_type = resource_type

    @property
    def event_is_optional(self):
        return self.provider.default_event == self

    @property
    def resource_is_optional(self):
        return self.resource_type in TriggerEvent.OPTIONAL_RESOURCE_TYPES