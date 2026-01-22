import copy
import logging
from s3transfer.utils import get_callbacks
def _log_and_set_exception(self, exception):
    logger.debug('Exception raised.', exc_info=True)
    self._transfer_coordinator.set_exception(exception)