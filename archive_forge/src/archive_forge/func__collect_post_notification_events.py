import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, on_trait_change
def _collect_post_notification_events(self, *args, **kwargs):
    self.post_change_events.append(args)
    self.exceptions.extend(kwargs.values())