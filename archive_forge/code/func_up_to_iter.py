from io import IOBase, TextIOWrapper
from typing import Iterable, Any, Union, Type, Optional
def up_to_iter(self, size: int) -> Iterable[Union[bytes, Any]]:
    """
        Yield up to size bytes from the iterator.
        """
    while size:
        if self.offset == len(self.chunk):
            try:
                self.chunk = next(self.iterator)
            except StopIteration:
                break
            else:
                self.offset = 0
        to_yield = min(size, len(self.chunk) - self.offset)
        self.offset = self.offset + to_yield
        size -= to_yield
        yield self.chunk[self.offset - to_yield:self.offset]