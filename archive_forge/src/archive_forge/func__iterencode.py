import re
def _iterencode(o, _current_indent_level):
    if isinstance(o, str):
        yield _encoder(o)
    elif o is None:
        yield 'null'
    elif o is True:
        yield 'true'
    elif o is False:
        yield 'false'
    elif isinstance(o, int):
        yield _intstr(o)
    elif isinstance(o, float):
        yield _floatstr(o)
    elif isinstance(o, (list, tuple)):
        yield from _iterencode_list(o, _current_indent_level)
    elif isinstance(o, dict):
        yield from _iterencode_dict(o, _current_indent_level)
    else:
        if markers is not None:
            markerid = id(o)
            if markerid in markers:
                raise ValueError('Circular reference detected')
            markers[markerid] = o
        o = _default(o)
        yield from _iterencode(o, _current_indent_level)
        if markers is not None:
            del markers[markerid]