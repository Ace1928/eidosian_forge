from typing import Any, Dict, Tuple
from langchain.chains.query_constructor.ir import (
def _map_comparator(self, comparator: Comparator) -> str:
    """
        Maps Langchain comparator to PostgREST comparator:

        https://postgrest.org/en/stable/references/api/tables_views.html#operators
        """
    postgrest_comparator = {Comparator.EQ: 'eq', Comparator.NE: 'neq', Comparator.GT: 'gt', Comparator.GTE: 'gte', Comparator.LT: 'lt', Comparator.LTE: 'lte', Comparator.LIKE: 'like'}.get(comparator)
    if postgrest_comparator is None:
        raise Exception(f"Comparator '{comparator}' is not currently supported in Supabase Vector")
    return postgrest_comparator