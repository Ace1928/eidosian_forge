import copy
import logging
from collections import deque, namedtuple
from botocore.compat import accepts_kwargs
from botocore.utils import EVENT_ALIASES
def _verify_and_register(self, event_name, handler, unique_id, register_method, unique_id_uses_count):
    self._verify_is_callable(handler)
    self._verify_accept_kwargs(handler)
    register_method(event_name, handler, unique_id, unique_id_uses_count)