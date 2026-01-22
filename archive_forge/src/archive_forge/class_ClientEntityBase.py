from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable
class ClientEntityBase:
    _client: Client
    max_per_page: int = 50

    def __init__(self, client: Client):
        """
        :param client: Client
        :return self
        """
        self._client = client

    def _iter_pages(self, list_function: Callable, *args, **kwargs) -> list:
        results = []
        page = 1
        while page:
            result, meta = list_function(*args, page=page, per_page=self.max_per_page, **kwargs)
            if result:
                results.extend(result)
            if meta and meta.pagination and meta.pagination.next_page:
                page = meta.pagination.next_page
            else:
                page = 0
        return results

    def _get_first_by(self, **kwargs):
        assert hasattr(self, 'get_list')
        entities, _ = self.get_list(**kwargs)
        return entities[0] if entities else None