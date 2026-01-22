import copy
import logging
from collections import deque, namedtuple
from botocore.compat import accepts_kwargs
from botocore.utils import EVENT_ALIASES
def _register_section(self, event_name, handler, unique_id, unique_id_uses_count, section):
    if unique_id is not None:
        if unique_id in self._unique_id_handlers:
            count = self._unique_id_handlers[unique_id].get('count', None)
            if unique_id_uses_count:
                if not count:
                    raise ValueError('Initial registration of  unique id %s was specified to use a counter. Subsequent register calls to unique id must specify use of a counter as well.' % unique_id)
                else:
                    self._unique_id_handlers[unique_id]['count'] += 1
            elif count:
                raise ValueError('Initial registration of unique id %s was specified to not use a counter. Subsequent register calls to unique id must specify not to use a counter as well.' % unique_id)
            return
        else:
            self._handlers.append_item(event_name, handler, section=section)
            unique_id_handler_item = {'handler': handler}
            if unique_id_uses_count:
                unique_id_handler_item['count'] = 1
            self._unique_id_handlers[unique_id] = unique_id_handler_item
    else:
        self._handlers.append_item(event_name, handler, section=section)
    self._lookup_cache = {}