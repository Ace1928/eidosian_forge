import traceback
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
class Registry(object):
    """Provides a registry for saving objects."""
    __slots__ = ['_name', '_registry']

    def __init__(self, name):
        """Creates a new registry."""
        self._name = name
        self._registry = {}

    def register(self, candidate, name=None):
        """Registers a Python object "candidate" for the given "name".

    Args:
      candidate: The candidate object to add to the registry.
      name: An optional string specifying the registry key for the candidate.
            If None, candidate.__name__ will be used.
    Raises:
      KeyError: If same name is used twice.
    """
        if not name:
            name = candidate.__name__
        if name in self._registry:
            frame = self._registry[name][_LOCATION_TAG]
            raise KeyError("Registering two %s with name '%s'! (Previous registration was in %s %s:%d)" % (self._name, name, frame.name, frame.filename, frame.lineno))
        logging.vlog(1, 'Registering %s (%s) in %s.', name, candidate, self._name)
        stack = traceback.extract_stack(limit=3)
        stack_index = min(2, len(stack) - 1)
        if stack_index >= 0:
            location_tag = stack[stack_index]
        else:
            location_tag = ('UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN')
        self._registry[name] = {_TYPE_TAG: candidate, _LOCATION_TAG: location_tag}

    def list(self):
        """Lists registered items.

    Returns:
      A list of names of registered objects.
    """
        return self._registry.keys()

    def lookup(self, name):
        """Looks up "name".

    Args:
      name: a string specifying the registry key for the candidate.
    Returns:
      Registered object if found
    Raises:
      LookupError: if "name" has not been registered.
    """
        name = compat.as_str(name)
        if name in self._registry:
            return self._registry[name][_TYPE_TAG]
        else:
            raise LookupError('%s registry has no entry for: %s' % (self._name, name))