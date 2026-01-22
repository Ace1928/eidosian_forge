from abc import abstractmethod
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Tuple, Union
from ._utils import StrByteType, StreamType
class PdfWriterProtocol(PdfCommonDocProtocol, Protocol):
    _objects: List[Any]
    _id_translated: Dict[int, Dict[int, int]]

    @abstractmethod
    def write(self, stream: Union[Path, StrByteType]) -> Tuple[bool, IO[Any]]:
        ...

    @abstractmethod
    def _add_object(self, obj: Any) -> Any:
        ...