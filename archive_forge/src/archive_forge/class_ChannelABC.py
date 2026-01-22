import abc
class ChannelABC(metaclass=abc.ABCMeta):
    """A base class for all channel ABCs."""

    @abc.abstractmethod
    def start(self) -> None:
        """Start the channel."""
        pass

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop the channel."""
        pass

    @abc.abstractmethod
    def is_alive(self) -> bool:
        """Test whether the channel is alive."""
        pass