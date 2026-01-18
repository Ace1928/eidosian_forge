from typing import Dict, Tuple, Union
from langchain.chains.query_constructor.ir import (
def visit_structured_query(self, structured_query: StructuredQuery) -> Tuple[str, dict]:
    if structured_query.filter is None:
        kwargs = {}
    else:
        kwargs = {'pre_filter': structured_query.filter.accept(self)}
    return (structured_query.query, kwargs)