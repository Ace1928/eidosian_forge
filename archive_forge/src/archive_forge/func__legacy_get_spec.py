import _imp
import _io
import sys
import _warnings
import marshal
@classmethod
def _legacy_get_spec(cls, fullname, finder):
    if hasattr(finder, 'find_loader'):
        msg = f'{_bootstrap._object_name(finder)}.find_spec() not found; falling back to find_loader()'
        _warnings.warn(msg, ImportWarning)
        loader, portions = finder.find_loader(fullname)
    else:
        msg = f'{_bootstrap._object_name(finder)}.find_spec() not found; falling back to find_module()'
        _warnings.warn(msg, ImportWarning)
        loader = finder.find_module(fullname)
        portions = []
    if loader is not None:
        return _bootstrap.spec_from_loader(fullname, loader)
    spec = _bootstrap.ModuleSpec(fullname, None)
    spec.submodule_search_locations = portions
    return spec