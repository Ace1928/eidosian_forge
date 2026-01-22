from asyncio import Future, Queue, ensure_future, sleep
from inspect import isawaitable
from typing import Any, AsyncIterator, Callable, Optional, Set
class SimplePubSub:
    """A very simple publish-subscript system.

    Creates an AsyncIterator from an EventEmitter.

    Useful for mocking a PubSub system for tests.
    """
    subscribers: Set[Callable]

    def __init__(self) -> None:
        self.subscribers = set()

    def emit(self, event: Any) -> bool:
        """Emit an event."""
        for subscriber in self.subscribers:
            result = subscriber(event)
            if isawaitable(result):
                ensure_future(result)
        return bool(self.subscribers)

    def get_subscriber(self, transform: Optional[Callable]=None) -> 'SimplePubSubIterator':
        return SimplePubSubIterator(self, transform)