import sys
def unindent_string(docstring):
    """ Set docstring to minimum indent for all lines, including first

    >>> unindent_string(' two')
    'two'
    >>> unindent_string('  two\\n   three')
    'two\\n three'
    """
    lines = docstring.expandtabs().splitlines()
    icount = indentcount_lines(lines)
    if icount == 0:
        return docstring
    return '\n'.join([line[icount:] for line in lines])