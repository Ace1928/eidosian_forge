import abc
from neutron_lib.api.definitions import portbindings
@property
@abc.abstractmethod
def top_bound_segment(self):
    """Return the current top-level bound segment dictionary.

        This property returns the current top-level bound segment
        dictionary, or None if the port is unbound. For a bound port,
        top_bound_segment is equivalent to
        binding_levels[0][BOUND_SEGMENT], and returns one of the
        port's network's static segments.
        """