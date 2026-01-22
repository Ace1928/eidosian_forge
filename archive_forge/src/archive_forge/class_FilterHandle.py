from contextlib import contextmanager
from itertools import count
from jeepney import HeaderFields, Message, MessageFlag, MessageType
class FilterHandle:

    def __init__(self, filters: MessageFilters, rule, queue):
        self._filters = filters
        self._filter_id = next(filters.filter_ids)
        self.rule = rule
        self.queue = queue
        self._filters.filters[self._filter_id] = self

    def close(self):
        del self._filters.filters[self._filter_id]

    def __enter__(self):
        return self.queue

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False