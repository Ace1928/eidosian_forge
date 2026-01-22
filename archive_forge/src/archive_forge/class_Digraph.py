from . import lang, files
class Digraph(Dot):
    """Directed graph source code in the DOT language."""
    if Graph.__doc__ is not None:
        __doc__ += Graph.__doc__.partition('.')[2]
    _head = 'digraph %s{'
    _head_strict = 'strict %s' % _head
    _edge = '\t%s -> %s%s'
    _edge_plain = _edge % ('%s', '%s', '')

    @property
    def directed(self):
        """``True``"""
        return True