from ._base import *
import operator as op
@dataclass
class LazyDBSaveMetrics:
    created: Optional[str] = None
    num_saved: Optional[int] = 0
    num_loaded: Optional[int] = 0
    last_load: Optional[str] = None
    last_save: Optional[str] = None
    time_alive: Optional[float] = 0