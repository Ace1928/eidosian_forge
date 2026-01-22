from ..utils import DummyObject, requires_backends
class Pix2StructImageProcessor(metaclass=DummyObject):
    _backends = ['vision']

    def __init__(self, *args, **kwargs):
        requires_backends(self, ['vision'])