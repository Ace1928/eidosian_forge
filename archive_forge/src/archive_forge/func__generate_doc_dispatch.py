from ._gi import \
def _generate_doc_dispatch(info):
    if isinstance(info, (ObjectInfo, StructInfo)):
        return _generate_class_info_doc(info)
    elif isinstance(info, CallableInfo):
        return _generate_callable_info_doc(info)
    return ''