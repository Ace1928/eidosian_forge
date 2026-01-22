from fontTools.misc.textTools import tostr
def _iterwalk(element, events, tag):
    include = tag is None or element.tag == tag
    if include and 'start' in events:
        yield ('start', element)
    for e in element:
        for item in _iterwalk(e, events, tag):
            yield item
    if include:
        yield ('end', element)