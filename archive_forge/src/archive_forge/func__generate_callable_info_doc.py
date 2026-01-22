from ._gi import \
def _generate_callable_info_doc(info):
    in_args_strs = []
    if isinstance(info, VFuncInfo):
        in_args_strs = ['self']
    elif isinstance(info, FunctionInfo):
        if info.is_method():
            in_args_strs = ['self']
    args = info.get_arguments()
    hint_blacklist = ('void',)
    ignore_indices = {info.get_return_type().get_array_length()}
    user_data_indices = set()
    for arg in args:
        ignore_indices.add(arg.get_destroy())
        ignore_indices.add(arg.get_type().get_array_length())
        user_data_indices.add(arg.get_closure())
    for i, arg in enumerate(args):
        if arg.get_direction() == Direction.OUT:
            continue
        if i in ignore_indices:
            continue
        argstr = arg.get_name()
        hint = _get_pytype_hint(arg.get_type())
        if hint not in hint_blacklist:
            argstr += ':' + hint
        if arg.may_be_null() or i in user_data_indices:
            argstr += '=None'
        elif arg.is_optional():
            argstr += '=<optional>'
        in_args_strs.append(argstr)
    in_args_str = ', '.join(in_args_strs)
    out_args_strs = []
    return_hint = _get_pytype_hint(info.get_return_type())
    if not info.skip_return() and return_hint and (return_hint not in hint_blacklist):
        argstr = return_hint
        if info.may_return_null():
            argstr += ' or None'
        out_args_strs.append(argstr)
    for i, arg in enumerate(args):
        if arg.get_direction() == Direction.IN:
            continue
        if i in ignore_indices:
            continue
        argstr = arg.get_name()
        hint = _get_pytype_hint(arg.get_type())
        if hint not in hint_blacklist:
            argstr += ':' + hint
        out_args_strs.append(argstr)
    if out_args_strs:
        return '%s(%s) -> %s' % (info.__name__, in_args_str, ', '.join(out_args_strs))
    else:
        return '%s(%s)' % (info.__name__, in_args_str)