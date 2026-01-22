from uuid import uuid4, UUID
from .parameter import Parameter
class ParameterVectorElement(Parameter):
    """An element of a ParameterVector."""
    ___slots__ = ('_vector', '_index')

    def __init__(self, vector, index, uuid=None):
        super().__init__(f'{vector.name}[{index}]', uuid=uuid)
        self._vector = vector
        self._index = index

    @property
    def index(self):
        """Get the index of this element in the parent vector."""
        return self._index

    @property
    def vector(self):
        """Get the parent vector instance."""
        return self._vector

    def __getstate__(self):
        return super().__getstate__() + (self._vector, self._index)

    def __setstate__(self, state):
        *super_state, vector, index = state
        super().__setstate__(super_state)
        self._vector = vector
        self._index = index