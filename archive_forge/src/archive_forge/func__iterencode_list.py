import re
def _iterencode_list(lst, _current_indent_level):
    if not lst:
        yield '[]'
        return
    if markers is not None:
        markerid = id(lst)
        if markerid in markers:
            raise ValueError('Circular reference detected')
        markers[markerid] = lst
    buf = '['
    if _indent is not None:
        _current_indent_level += 1
        newline_indent = '\n' + _indent * _current_indent_level
        separator = _item_separator + newline_indent
        buf += newline_indent
    else:
        newline_indent = None
        separator = _item_separator
    first = True
    for value in lst:
        if first:
            first = False
        else:
            buf = separator
        if isinstance(value, str):
            yield (buf + _encoder(value))
        elif value is None:
            yield (buf + 'null')
        elif value is True:
            yield (buf + 'true')
        elif value is False:
            yield (buf + 'false')
        elif isinstance(value, int):
            yield (buf + _intstr(value))
        elif isinstance(value, float):
            yield (buf + _floatstr(value))
        else:
            yield buf
            if isinstance(value, (list, tuple)):
                chunks = _iterencode_list(value, _current_indent_level)
            elif isinstance(value, dict):
                chunks = _iterencode_dict(value, _current_indent_level)
            else:
                chunks = _iterencode(value, _current_indent_level)
            yield from chunks
    if newline_indent is not None:
        _current_indent_level -= 1
        yield ('\n' + _indent * _current_indent_level)
    yield ']'
    if markers is not None:
        del markers[markerid]