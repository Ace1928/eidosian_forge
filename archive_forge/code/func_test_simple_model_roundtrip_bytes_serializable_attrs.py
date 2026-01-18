import pytest
import srsly
from thinc.api import (
def test_simple_model_roundtrip_bytes_serializable_attrs():
    fwd = lambda model, X, is_train: (X, lambda dY: dY)
    attr = SerializableAttr()
    assert attr.value == 'foo'
    assert attr.to_bytes() == b'foo'
    model = Model('test', fwd, attrs={'test': attr})
    model.initialize()

    @serialize_attr.register(SerializableAttr)
    def serialize_attr_custom(_, value, name, model):
        return value.to_bytes()

    @deserialize_attr.register(SerializableAttr)
    def deserialize_attr_custom(_, value, name, model):
        return SerializableAttr().from_bytes(value)
    model_bytes = model.to_bytes()
    model = model.from_bytes(model_bytes)
    assert 'test' in model.attrs
    assert model.attrs['test'].value == 'foo from bytes'