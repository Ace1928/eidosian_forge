from django.conf import settings
from django.contrib.messages import constants, utils
from django.utils.functional import SimpleLazyObject
def _get_level(self):
    """
        Return the minimum recorded level.

        The default level is the ``MESSAGE_LEVEL`` setting. If this is
        not found, the ``INFO`` level is used.
        """
    if not hasattr(self, '_level'):
        self._level = getattr(settings, 'MESSAGE_LEVEL', constants.INFO)
    return self._level