from gast import AST  # so that metadata are walkable as regular ast nodes
class Local(AST):
    """ Metadata to mark function as non exported. """