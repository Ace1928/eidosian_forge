from typing import List, Union
def with_schema(self) -> 'AggregateRequest':
    """
        If set, the `schema` property will contain a list of `[field, type]`
        entries in the result object.
        """
    self._with_schema = True
    return self