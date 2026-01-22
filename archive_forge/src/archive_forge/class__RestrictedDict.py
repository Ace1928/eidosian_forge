from io import StringIO
import numbers
from typing import (
from Bio import BiopythonDeprecationWarning, StreamModeError
from Bio.Seq import Seq, MutableSeq, UndefinedSequenceError
import warnings
class _RestrictedDict(Dict[str, Sequence[Any]]):
    """Dict which only allows sequences of given length as values (PRIVATE).

    This simple subclass of the Python dictionary is used in the SeqRecord
    object for holding per-letter-annotations.  This class is intended to
    prevent simple errors by only allowing python sequences (e.g. lists,
    strings and tuples) to be stored, and only if their length matches that
    expected (the length of the SeqRecord's seq object).  It cannot however
    prevent the entries being edited in situ (for example appending entries
    to a list).

    >>> x = _RestrictedDict(5)
    >>> x["test"] = "hello"
    >>> x
    {'test': 'hello'}

    Adding entries which don't have the expected length are blocked:

    >>> x["test"] = "hello world"
    Traceback (most recent call last):
    ...
    TypeError: Any per-letter annotation should be a Python sequence (list, tuple or string) of the same length as the biological sequence, here 5.

    The expected length is stored as a private attribute,

    >>> x._length
    5

    In order that the SeqRecord (and other objects using this class) can be
    pickled, for example for use in the multiprocessing library, we need to
    be able to pickle the restricted dictionary objects.

    Using the default protocol, which is 3 on Python 3,

    >>> import pickle
    >>> y = pickle.loads(pickle.dumps(x))
    >>> y
    {'test': 'hello'}
    >>> y._length
    5

    Using the highest protocol, which is 4 on Python 3,

    >>> import pickle
    >>> z = pickle.loads(pickle.dumps(x, pickle.HIGHEST_PROTOCOL))
    >>> z
    {'test': 'hello'}
    >>> z._length
    5
    """

    def __init__(self, length: int) -> None:
        """Create an EMPTY restricted dictionary."""
        dict.__init__(self)
        self._length = int(length)

    def __setitem__(self, key: str, value: Sequence[Any]) -> None:
        if not hasattr(value, '__len__') or not hasattr(value, '__getitem__') or (hasattr(self, '_length') and len(value) != self._length):
            raise TypeError(f'Any per-letter annotation should be a Python sequence (list, tuple or string) of the same length as the biological sequence, here {self._length}.')
        dict.__setitem__(self, key, value)

    def update(self, new_dict):
        for key, value in new_dict.items():
            self[key] = value