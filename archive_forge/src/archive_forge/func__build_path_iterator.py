import re
def _build_path_iterator(path, namespaces, with_prefixes=True):
    """compile selector pattern"""
    if path[-1:] == '/':
        path += '*'
    cache_key = (path,)
    if namespaces:
        if None in namespaces:
            if '' in namespaces and namespaces[None] != namespaces['']:
                raise ValueError('Ambiguous default namespace provided: %r versus %r' % (namespaces[None], namespaces['']))
            cache_key += (namespaces[None],) + tuple(sorted((item for item in namespaces.items() if item[0] is not None)))
        else:
            cache_key += tuple(sorted(namespaces.items()))
    try:
        return _cache[cache_key]
    except KeyError:
        pass
    if len(_cache) > 100:
        _cache.clear()
    if path[:1] == '/':
        raise SyntaxError('cannot use absolute path on element')
    stream = iter(xpath_tokenizer(path, namespaces, with_prefixes=with_prefixes))
    try:
        _next = stream.next
    except AttributeError:
        _next = stream.__next__
    try:
        token = _next()
    except StopIteration:
        raise SyntaxError('empty path expression')
    selector = []
    while 1:
        try:
            selector.append(ops[token[0]](_next, token))
        except StopIteration:
            raise SyntaxError('invalid path')
        try:
            token = _next()
            if token[0] == '/':
                token = _next()
        except StopIteration:
            break
    _cache[cache_key] = selector
    return selector