import operator
import warnings
def to_tuple_or_list(value):
    """Convert the non-`None` value to either a tuple or a list."""
    if value is None:
        return value
    if not isinstance(value, (int, tuple, list)):
        raise ValueError(f'`value` must be an integer, tuple or list. Received: value={value}')
    if isinstance(value, int):
        return (value,)
    return value