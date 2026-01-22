from typing import List, Union
def alias(self, alias: str) -> 'Reducer':
    """
        Set the alias for this reducer.

        ### Parameters

        - **alias**: The value of the alias for this reducer. If this is the
            special value `aggregation.FIELDNAME` then this reducer will be
            aliased using the same name as the field upon which it operates.
            Note that using `FIELDNAME` is only possible on reducers which
            operate on a single field value.

        This method returns the `Reducer` object making it suitable for
        chaining.
        """
    if alias is FIELDNAME:
        if not self._field:
            raise ValueError('Cannot use FIELDNAME alias with no field')
        alias = self._field[1:]
    self._alias = alias
    return self