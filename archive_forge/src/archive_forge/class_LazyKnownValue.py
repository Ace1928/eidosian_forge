from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.common import monkeypatch
class LazyKnownValue(AbstractLazyValue):
    """data is a Value."""

    def infer(self):
        return ValueSet([self.data])