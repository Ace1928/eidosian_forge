import abc
import enum
from taskflow import atom
from taskflow import exceptions as exc
from taskflow.utils import misc
class ParameterizedForEach(ForEachBase):
    """Applies a dynamically provided collection of strategies.

    Accepts a collection of decision strategies from a predecessor (or from
    storage) as a parameter and returns the next element of that collection on
    each try.

    :param revert_all: when provided this will cause the full flow to revert
                       when the number of attempts that have been tried
                       has been reached (when false, it will only locally
                       revert the associated subflow)
    :type revert_all: bool

    Further arguments are interpreted as defined in the
    :py:class:`~taskflow.atom.Atom` constructor.
    """

    def __init__(self, name=None, provides=None, requires=None, auto_extract=True, rebind=None, revert_all=False):
        super(ParameterizedForEach, self).__init__(name, provides, requires, auto_extract, rebind, revert_all)

    def on_failure(self, values, history, *args, **kwargs):
        return self._on_failure(values, history)

    def execute(self, values, history, *args, **kwargs):
        return self._get_next_value(values, history)