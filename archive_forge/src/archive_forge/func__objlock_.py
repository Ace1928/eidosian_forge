import copy
import threading
import contextlib
import operator
import copyreg
from typing import Any, Type, Tuple, Dict, List, Union, Optional, Callable, TypeVar, Iterable, Generic, TYPE_CHECKING
@contextlib.contextmanager
def _objlock_(self):
    """
        Returns the object lock
        """
    if self.__dict__['__threadlock_'] is not None:
        try:
            with self.__dict__['__threadlock_']:
                yield
        except Exception as e:
            raise e
    else:
        yield