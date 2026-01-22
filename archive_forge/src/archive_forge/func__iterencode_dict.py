import re
def _iterencode_dict(dct, _current_indent_level):
    if not dct:
        yield '{}'
        return
    if markers is not None:
        markerid = id(dct)
        if markerid in markers:
            raise ValueError('Circular reference detected')
        markers[markerid] = dct
    yield '{'
    if _indent is not None:
        _current_indent_level += 1
        newline_indent = '\n' + _indent * _current_indent_level
        item_separator = _item_separator + newline_indent
        yield newline_indent
    else:
        newline_indent = None
        item_separator = _item_separator
    first = True
    if _sort_keys:
        items = sorted(dct.items())
    else:
        items = dct.items()
    for key, value in items:
        if isinstance(key, str):
            pass
        elif isinstance(key, float):
            key = _floatstr(key)
        elif key is True:
            key = 'true'
        elif key is False:
            key = 'false'
        elif key is None:
            key = 'null'
        elif isinstance(key, int):
            key = _intstr(key)
        elif _skipkeys:
            continue
        else:
            raise TypeError(f'keys must be str, int, float, bool or None, not {key.__class__.__name__}')
        if first:
            first = False
        else:
            yield item_separator
        yield _encoder(key)
        yield _key_separator
        if isinstance(value, str):
            yield _encoder(value)
        elif value is None:
            yield 'null'
        elif value is True:
            yield 'true'
        elif value is False:
            yield 'false'
        elif isinstance(value, int):
            yield _intstr(value)
        elif isinstance(value, float):
            yield _floatstr(value)
        else:
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
    yield '}'
    if markers is not None:
        del markers[markerid]