from ._base import *
class LazyIOMultiFile:

    def __init__(self, filenames: List[LazyIOType], mode: ModeIO=ModeIO.read):
        self._multi_io = []
        for filename in filenames:
            if isinstance(filename, str):
                pass
        pass