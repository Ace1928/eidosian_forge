from typing import Generic, TypeVar, AsyncContextManager
from abc import abstractmethod, ABCMeta
class ConnectionFactory(Generic[Request, Response], metaclass=ABCMeta):
    """A factory for producing Connections."""

    @abstractmethod
    async def new(self) -> Connection[Request, Response]:
        raise NotImplementedError()