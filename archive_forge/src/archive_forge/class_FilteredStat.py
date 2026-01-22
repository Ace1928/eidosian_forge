from io import BytesIO
from typing import Callable, Dict, List, Tuple
from .. import errors, osutils, registry
class FilteredStat:

    def __init__(self, base, st_size=None):
        self.st_mode = base.st_mode
        self.st_size = st_size or base.st_size
        self.st_mtime = base.st_mtime
        self.st_ctime = base.st_ctime