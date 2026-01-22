import os
import warnings
from enchant.errors import Error, DictNotFoundError
from enchant.utils import get_default_language
from enchant.pypwl import PyPWL
def _init_this(self):
    self._this = None
    if self.tag:
        this = self._broker._request_dict_data(self.tag)
        self._switch_this(this, self._broker)