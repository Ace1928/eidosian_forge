import html
from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar
from triad import ParamDict, SerializableRLock, assert_or_throw
from .._utils.registry import fugue_plugin
from ..exceptions import FugueDatasetEmptyError
class DatasetDisplay(ABC):
    """The base class for display handlers of :class:`~.Dataset`

    :param ds: the Dataset
    """
    _SHOW_LOCK = SerializableRLock()

    def __init__(self, ds: Dataset):
        self._ds = ds

    @abstractmethod
    def show(self, n: int=10, with_count: bool=False, title: Optional[str]=None) -> None:
        """Show the :class:`~.Dataset`

        :param n: top n items to display, defaults to 10
        :param with_count: whether to display the total count, defaults to False
        :param title: title to display, defaults to None
        """
        raise NotImplementedError

    def repr(self) -> str:
        """The string representation of the :class:`~.Dataset`

        :return: the string representation
        """
        return str(type(self._ds).__name__)

    def repr_html(self) -> str:
        """The HTML representation of the :class:`~.Dataset`

        :return: the HTML representation
        """
        return html.escape(self.repr())