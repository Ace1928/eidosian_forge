from .. import axes, _docstring
from .geo import AitoffAxes, HammerAxes, LambertAxes, MollweideAxes
from .polar import PolarAxes
class ProjectionRegistry:
    """A mapping of registered projection names to projection classes."""

    def __init__(self):
        self._all_projection_types = {}

    def register(self, *projections):
        """Register a new set of projections."""
        for projection in projections:
            name = projection.name
            self._all_projection_types[name] = projection

    def get_projection_class(self, name):
        """Get a projection class from its *name*."""
        return self._all_projection_types[name]

    def get_projection_names(self):
        """Return the names of all projections currently registered."""
        return sorted(self._all_projection_types)