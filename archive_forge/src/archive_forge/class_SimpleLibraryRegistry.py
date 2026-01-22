from .abstract_impl import AbstractImplHolder
class SimpleLibraryRegistry:
    """Registry for the "simple" torch.library APIs

    The "simple" torch.library APIs are a higher-level API on top of the
    raw PyTorch DispatchKey registration APIs that includes:
    - abstract impl

    Registrations for these APIs do not go into the PyTorch dispatcher's
    table because they may not directly involve a DispatchKey. For example,
    the abstract impl is a Python function that gets invoked by FakeTensor.
    Instead, we manage them here.

    SimpleLibraryRegistry is a mapping from a fully qualified operator name
    (including the overload) to SimpleOperatorEntry.
    """

    def __init__(self):
        self._data = {}

    def find(self, qualname: str) -> 'SimpleOperatorEntry':
        if qualname not in self._data:
            self._data[qualname] = SimpleOperatorEntry(qualname)
        return self._data[qualname]