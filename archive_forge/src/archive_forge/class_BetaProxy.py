from typing_extensions import override
from . import resources, _load_client
from ._utils import LazyProxy
class BetaProxy(LazyProxy[resources.Beta]):

    @override
    def __load__(self) -> resources.Beta:
        return _load_client().beta