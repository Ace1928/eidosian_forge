import torch
class _ReturnTypeCM:

    def __init__(self, to_restore):
        self.to_restore = to_restore

    def __enter__(self):
        return self

    def __exit__(self, *args):
        global _TORCHFUNCTION_SUBCLASS
        _TORCHFUNCTION_SUBCLASS = self.to_restore