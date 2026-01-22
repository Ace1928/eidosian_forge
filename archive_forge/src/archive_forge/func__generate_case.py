import inspect
import sys
def _generate_case(base, module, cls_name, mb, method_generator):
    members = mb.copy()
    base_methods = inspect.getmembers(base, predicate=inspect.isfunction)
    for name, value in base_methods:
        if not name.startswith('test_'):
            continue
        value = method_generator(value)
        members[name] = value
    cls = type(cls_name, (base,), members)
    cls.__module__ = module.__name__
    setattr(module, cls_name, cls)