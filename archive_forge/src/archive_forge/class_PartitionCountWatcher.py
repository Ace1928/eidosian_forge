from abc import abstractmethod, ABCMeta
from typing import AsyncContextManager
class PartitionCountWatcher(AsyncContextManager, metaclass=ABCMeta):

    @abstractmethod
    async def get_partition_count(self) -> int:
        raise NotImplementedError()