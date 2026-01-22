from ..overrides import override
from ..module import get_introspection_module
class OverridesObject(GIMarshallingTests.OverridesObject):

    def __new__(cls, long_):
        return GIMarshallingTests.OverridesObject.__new__(cls)

    def __init__(self, long_):
        GIMarshallingTests.OverridesObject.__init__(self)

    @classmethod
    def new(cls, long_):
        self = GIMarshallingTests.OverridesObject.new()
        return self

    def method(self):
        """Overridden doc string."""
        return GIMarshallingTests.OverridesObject.method(self) / 7