from abc import ABCMeta, abstractmethod
class AbstractOptionValue(metaclass=ABCMeta):
    """Abstract base class for custom option values.
    """

    @abstractmethod
    def encode(self) -> str:
        """Returns an encoding of the values
        """
        ...

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.encode()})'