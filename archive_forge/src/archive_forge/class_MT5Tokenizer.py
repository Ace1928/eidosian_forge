from ..utils import DummyObject, requires_backends
class MT5Tokenizer(metaclass=DummyObject):
    _backends = ['sentencepiece']

    def __init__(self, *args, **kwargs):
        requires_backends(self, ['sentencepiece'])