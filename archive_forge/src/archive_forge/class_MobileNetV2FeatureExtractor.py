from ..utils import DummyObject, requires_backends
class MobileNetV2FeatureExtractor(metaclass=DummyObject):
    _backends = ['vision']

    def __init__(self, *args, **kwargs):
        requires_backends(self, ['vision'])