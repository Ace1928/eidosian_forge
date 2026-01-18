import collections.abc
import io
import itertools
import types
import typing
Check that the left argument is a subtype of the right.
    For unions, check if the type arguments of the left is a subset of the right.
    Also works for nested types including ForwardRefs.

    Examples:

    ```python
        from typing_utils import issubtype

        issubtype(typing.List, typing.Any) == True
        issubtype(list, list) == True
        issubtype(list, typing.List) == True
        issubtype(list, typing.Sequence) == True
        issubtype(typing.List[int], list) == True
        issubtype(typing.List[typing.List], list) == True
        issubtype(list, typing.List[int]) == False
        issubtype(list, typing.Union[typing.Tuple, typing.Set]) == False
        issubtype(typing.List[typing.List], typing.List[typing.Sequence]) == True
        JSON = typing.Union[
            int, float, bool, str, None, typing.Sequence["JSON"],
            typing.Mapping[str, "JSON"]
        ]
        issubtype(str, JSON, forward_refs={'JSON': JSON}) == True
        issubtype(typing.Dict[str, str], JSON, forward_refs={'JSON': JSON}) == True
        issubtype(typing.Dict[str, bytes], JSON, forward_refs={'JSON': JSON}) == False
    ```
    