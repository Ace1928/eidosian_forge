from typing import FrozenSet, Optional, Set, Tuple
import gevent
from gevent import pool
from .object_store import (
from .objects import Commit, ObjectID, Tag
Find the objects missing from another object store.

    Same implementation as object_store.MissingObjectFinder
    except we use gevent to parallelize object retrieval.
    