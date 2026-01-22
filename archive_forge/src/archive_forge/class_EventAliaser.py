import copy
import logging
from collections import deque, namedtuple
from botocore.compat import accepts_kwargs
from botocore.utils import EVENT_ALIASES
class EventAliaser(BaseEventHooks):

    def __init__(self, event_emitter, event_aliases=None):
        self._event_aliases = event_aliases
        if event_aliases is None:
            self._event_aliases = EVENT_ALIASES
        self._alias_name_cache = {}
        self._emitter = event_emitter

    def emit(self, event_name, **kwargs):
        aliased_event_name = self._alias_event_name(event_name)
        return self._emitter.emit(aliased_event_name, **kwargs)

    def emit_until_response(self, event_name, **kwargs):
        aliased_event_name = self._alias_event_name(event_name)
        return self._emitter.emit_until_response(aliased_event_name, **kwargs)

    def register(self, event_name, handler, unique_id=None, unique_id_uses_count=False):
        aliased_event_name = self._alias_event_name(event_name)
        return self._emitter.register(aliased_event_name, handler, unique_id, unique_id_uses_count)

    def register_first(self, event_name, handler, unique_id=None, unique_id_uses_count=False):
        aliased_event_name = self._alias_event_name(event_name)
        return self._emitter.register_first(aliased_event_name, handler, unique_id, unique_id_uses_count)

    def register_last(self, event_name, handler, unique_id=None, unique_id_uses_count=False):
        aliased_event_name = self._alias_event_name(event_name)
        return self._emitter.register_last(aliased_event_name, handler, unique_id, unique_id_uses_count)

    def unregister(self, event_name, handler=None, unique_id=None, unique_id_uses_count=False):
        aliased_event_name = self._alias_event_name(event_name)
        return self._emitter.unregister(aliased_event_name, handler, unique_id, unique_id_uses_count)

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

    def _replace_subsection(self, sections, old_parts, new_part):
        for i in range(len(sections)):
            if sections[i] == old_parts[0] and sections[i:i + len(old_parts)] == old_parts:
                sections[i:i + len(old_parts)] = [new_part]
                return

    def __copy__(self):
        return self.__class__(copy.copy(self._emitter), copy.copy(self._event_aliases))