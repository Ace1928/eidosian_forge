from ..utils import DummyObject, requires_backends
class TFMT5Model(metaclass=DummyObject):
    _backends = ['tf']

    def __init__(self, *args, **kwargs):
        requires_backends(self, ['tf'])