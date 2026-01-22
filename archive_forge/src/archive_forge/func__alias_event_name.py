import copy
import logging
from collections import deque, namedtuple
from botocore.compat import accepts_kwargs
from botocore.utils import EVENT_ALIASES
def _alias_event_name(self, event_name):
    if event_name in self._alias_name_cache:
        return self._alias_name_cache[event_name]
    for old_part, new_part in self._event_aliases.items():
        event_parts = event_name.split('.')
        if '.' not in old_part:
            try:
                event_parts[event_parts.index(old_part)] = new_part
            except ValueError:
                continue
        elif old_part in event_name:
            old_parts = old_part.split('.')
            self._replace_subsection(event_parts, old_parts, new_part)
        else:
            continue
        new_name = '.'.join(event_parts)
        logger.debug(f'Changing event name from {event_name} to {new_name}')
        self._alias_name_cache[event_name] = new_name
        return new_name
    self._alias_name_cache[event_name] = event_name
    return event_name