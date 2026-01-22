from .bindings.libnvpair import ffi as _ffi
def _ffi_cast(type_name):
    type_info = _ffi.typeof(type_name)

    def _func(value):
        if type_info.kind == 'enum':
            try:
                type_info.elements[value]
            except KeyError as e:
                raise OverflowError('Invalid enum <%s> value %s' % (type_info.cname, e.message))
        else:
            _ffi.new(type_name + '*', value)
        return _ffi.cast(type_name, value)
    _func.__name__ = type_name
    return _func