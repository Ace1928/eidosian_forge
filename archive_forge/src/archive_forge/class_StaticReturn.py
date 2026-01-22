from gast import AST  # so that metadata are walkable as regular ast nodes
class StaticReturn(AST):
    """ Metadata to mark return with a constant value. """