from typing_extensions import override
from . import resources, _load_client
from ._utils import LazyProxy
class FilesProxy(LazyProxy[resources.Files]):

    @override
    def __load__(self) -> resources.Files:
        return _load_client().files