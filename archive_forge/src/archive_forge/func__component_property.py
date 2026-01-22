from . import _check_status, cairo, ffi
def _component_property(name):
    return property(lambda self: getattr(self._pointer, name), lambda self, value: setattr(self._pointer, name, value), doc='Read-write attribute access to a single float component.')