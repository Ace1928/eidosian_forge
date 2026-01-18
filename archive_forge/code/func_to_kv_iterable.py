from typing import (
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import to_size
def to_kv_iterable(data: Any, none_as_empty: bool=True) -> Iterable[Tuple[Any, Any]]:
    """Convert data to iterable of key value pairs

    :param data: input object, it can be a dict or Iterable[Tuple[Any, Any]]
        or Iterable[List[Any]]
    :param none_as_empty: if to treat None as empty iterable

    :raises ValueError: if input is None and `none_as_empty==False`
    :raises ValueError: if input is a set
    :raises TypeError or ValueError: if input data type is not acceptable

    :yield: iterable of key value pair as tuples
    """
    if data is None:
        assert_or_throw(none_as_empty, ValueError("data can't be None"))
    elif isinstance(data, Dict):
        for k, v in data.items():
            yield (k, v)
    elif isinstance(data, Set):
        raise ValueError(f'{data} is a set, did you mistakenly use `,` instead of `:`?')
    elif isinstance(data, Iterable):
        ei = make_empty_aware(data)
        if not ei.empty:
            first = ei.peek()
            if isinstance(first, tuple):
                for k, v in ei:
                    yield (k, v)
            elif isinstance(first, List):
                for arr in ei:
                    if len(arr) == 2:
                        yield (arr[0], arr[1])
                    else:
                        raise TypeError(f'{arr} is not an acceptable item')
            else:
                raise TypeError(f'{first} is not an acceptable item')
    else:
        raise TypeError(f'{type(data)} is not supported')