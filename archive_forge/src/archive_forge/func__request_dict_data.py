import os
import warnings
from enchant.errors import Error, DictNotFoundError
from enchant.utils import get_default_language
from enchant.pypwl import PyPWL
def _request_dict_data(self, tag):
    """Request raw C pointer data for a dictionary.

        This method call passes on the call to the C library, and does
        some internal bookkeeping.
        """
    self._check_this()
    new_dict = _e.broker_request_dict(self._this, tag.encode())
    if new_dict is None:
        e_str = "Dictionary for language '%s' could not be found\n"
        e_str += 'Please check https://pyenchant.github.io/pyenchant/ for details'
        self._raise_error(e_str % (tag,), DictNotFoundError)
    if new_dict not in self._live_dicts:
        self._live_dicts[new_dict] = 1
    else:
        self._live_dicts[new_dict] += 1
    return new_dict