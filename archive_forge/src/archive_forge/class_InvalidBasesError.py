from django.db import DatabaseError
class InvalidBasesError(ValueError):
    """A model's base classes can't be resolved."""
    pass