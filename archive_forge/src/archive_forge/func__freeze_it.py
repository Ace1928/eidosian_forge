from taskflow.listeners import base
def _freeze_it(values):
    """Freezes a set of values (handling none/empty nicely)."""
    if not values:
        return frozenset()
    else:
        return frozenset(values)