from jedi import debug
from jedi.inference.base_value import ValueSet, \
from jedi.inference.utils import to_list
from jedi.inference.gradual.stub_value import StubModuleValue
from jedi.inference.gradual.typeshed import try_to_load_stub_cached
from jedi.inference.value.decorator import Decoratee
@to_list
def _python_to_stub_names(names, fallback_to_python=False):
    for name in names:
        module_context = name.get_root_context()
        if module_context.is_stub():
            yield name
            continue
        if name.api_type == 'module':
            found_name = False
            for n in name.goto():
                if n.api_type == 'module':
                    values = convert_values(n.infer(), only_stubs=True)
                    for v in values:
                        yield v.name
                        found_name = True
                else:
                    for x in _python_to_stub_names([n], fallback_to_python=fallback_to_python):
                        yield x
                        found_name = True
            if found_name:
                continue
        else:
            v = name.get_defining_qualified_value()
            if v is not None:
                converted = to_stub(v)
                if converted:
                    converted_names = converted.goto(name.get_public_name())
                    if converted_names:
                        yield from converted_names
                        continue
        if fallback_to_python:
            yield name