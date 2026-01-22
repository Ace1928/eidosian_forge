from typing_extensions import override
from . import resources, _load_client
from ._utils import LazyProxy
class FineTuningProxy(LazyProxy[resources.FineTuning]):

    @override
    def __load__(self) -> resources.FineTuning:
        return _load_client().fine_tuning