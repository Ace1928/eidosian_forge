from ._base import *
import operator as op
@lazyclass
@dataclass
class LazyDBConfig:
    dbschema: Dict[str, Any]
    autosave: bool = True
    autouser: bool = True
    is_dev: bool = True
    savefreq: float = 15.0
    seeddata: Optional[Dict[str, Any]] = None
    userconfigs: Optional[Dict[str, Any]] = None
    hashschema: Optional[Dict[str, Any]] = None
    dbname: Optional[str] = None