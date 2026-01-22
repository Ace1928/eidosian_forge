import zope.interface
def _trim_doc_string(text):
    """ Trims a doc string to make it format
    correctly with structured text. """
    lines = text.replace('\r\n', '\n').split('\n')
    nlines = [lines.pop(0)]
    if lines:
        min_indent = min([len(line) - len(line.lstrip()) for line in lines])
        for line in lines:
            nlines.append(line[min_indent:])
    return '\n'.join(nlines)