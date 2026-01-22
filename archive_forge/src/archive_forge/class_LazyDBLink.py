from ._base import *
import operator as op
class LazyDBLink:

    def __init__(self, name: str, index, method: Optional[LazyLink]=LazyLink.HAS, *args, **kwargs):
        pass

    def __ne__(self, o: object) -> bool:
        pass

    def __eq__(self, o: object) -> bool:
        pass

    def __ge__(self, o):
        pass

    def __gt__(self, o):
        pass

    def __lt__(self, o):
        pass

    def __le__(self, o):
        pass