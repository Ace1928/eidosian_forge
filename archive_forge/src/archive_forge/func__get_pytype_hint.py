from ._gi import \
def _get_pytype_hint(gi_type):
    type_tag = gi_type.get_tag()
    py_type = _type_tag_to_py_type.get(type_tag, None)
    if py_type and hasattr(py_type, '__name__'):
        return py_type.__name__
    elif type_tag == TypeTag.INTERFACE:
        iface = gi_type.get_interface()
        info_name = iface.get_name()
        if not info_name:
            return gi_type.get_tag_as_string()
        return '%s.%s' % (iface.get_namespace(), info_name)
    return gi_type.get_tag_as_string()