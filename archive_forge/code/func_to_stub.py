from jedi import debug
from jedi.inference.base_value import ValueSet, \
from jedi.inference.utils import to_list
from jedi.inference.gradual.stub_value import StubModuleValue
from jedi.inference.gradual.typeshed import try_to_load_stub_cached
from jedi.inference.value.decorator import Decoratee
def to_stub(value):
    if value.is_stub():
        return ValueSet([value])
    was_instance = value.is_instance()
    if was_instance:
        value = value.py__class__()
    qualified_names = value.get_qualified_names()
    stub_module = _load_stub_module(value.get_root_context().get_value())
    if stub_module is None or qualified_names is None:
        return NO_VALUES
    was_bound_method = value.is_bound_method()
    if was_bound_method:
        method_name = qualified_names[-1]
        qualified_names = qualified_names[:-1]
        was_instance = True
    stub_values = ValueSet([stub_module])
    for name in qualified_names:
        stub_values = stub_values.py__getattribute__(name)
    if was_instance:
        stub_values = ValueSet.from_sets((c.execute_with_values() for c in stub_values if c.is_class()))
    if was_bound_method:
        stub_values = stub_values.py__getattribute__(method_name)
    return stub_values