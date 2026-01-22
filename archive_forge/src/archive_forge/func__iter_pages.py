from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable
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