import copy
import logging
from s3transfer.utils import get_callbacks
def _wait_on_dependent_futures(self):
    futures_to_wait_on = []
    for _, future in self._pending_main_kwargs.items():
        if isinstance(future, list):
            futures_to_wait_on.extend(future)
        else:
            futures_to_wait_on.append(future)
    self._wait_until_all_complete(futures_to_wait_on)