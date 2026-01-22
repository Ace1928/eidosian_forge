from jedi import debug
from jedi.inference.base_value import ValueSet, \
from jedi.inference.utils import to_list
from jedi.inference.gradual.stub_value import StubModuleValue
from jedi.inference.gradual.typeshed import try_to_load_stub_cached
from jedi.inference.value.decorator import Decoratee
def _stub_to_python_value_set(stub_value, ignore_compiled=False):
    stub_module_context = stub_value.get_root_context()
    if not stub_module_context.is_stub():
        return ValueSet([stub_value])
    decorates = None
    if isinstance(stub_value, Decoratee):
        decorates = stub_value._original_value
    was_instance = stub_value.is_instance()
    if was_instance:
        arguments = getattr(stub_value, '_arguments', None)
        stub_value = stub_value.py__class__()
    qualified_names = stub_value.get_qualified_names()
    if qualified_names is None:
        return NO_VALUES
    was_bound_method = stub_value.is_bound_method()
    if was_bound_method:
        method_name = qualified_names[-1]
        qualified_names = qualified_names[:-1]
        was_instance = True
        arguments = None
    values = _infer_from_stub(stub_module_context, qualified_names, ignore_compiled)
    if was_instance:
        values = ValueSet.from_sets((c.execute_with_values() if arguments is None else c.execute(arguments) for c in values if c.is_class()))
    if was_bound_method:
        values = values.py__getattribute__(method_name)
    if decorates is not None:
        values = ValueSet((Decoratee(v, decorates) for v in values))
    return values