from contextlib import contextmanager
from kombu.utils.objects import cached_property
@cached_property
def Receiver(self):
    return self.app.subclass_with_self(self.receiver_cls, reverse='events.Receiver')