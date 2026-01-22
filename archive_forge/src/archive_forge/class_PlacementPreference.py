from .. import errors
from ..constants import IS_WINDOWS_PLATFORM
from ..utils import (
class PlacementPreference(dict):
    """
        Placement preference to be used as an element in the list of
        preferences for :py:class:`Placement` objects.

        Args:
            strategy (string): The placement strategy to implement. Currently,
                the only supported strategy is ``spread``.
            descriptor (string): A label descriptor. For the spread strategy,
                the scheduler will try to spread tasks evenly over groups of
                nodes identified by this label.
    """

    def __init__(self, strategy, descriptor):
        if strategy != 'spread':
            raise errors.InvalidArgument('PlacementPreference strategy value is invalid ({}): must be "spread".'.format(strategy))
        self['Spread'] = {'SpreadDescriptor': descriptor}