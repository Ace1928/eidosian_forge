import enum
def visitList(self, obj, *args, **kwargs):
    """Called to visit any value that is a list."""
    for value in obj:
        self.visit(value, *args, **kwargs)