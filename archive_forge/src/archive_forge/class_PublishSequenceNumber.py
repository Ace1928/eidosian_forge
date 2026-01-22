from typing import NamedTuple
class PublishSequenceNumber(NamedTuple):
    value: int

    def next(self) -> 'PublishSequenceNumber':
        return PublishSequenceNumber(self.value + 1)