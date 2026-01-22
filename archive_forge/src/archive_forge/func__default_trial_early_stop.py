from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
from triad import SerializableRLock, to_uuid
from tune.concepts.flow import (
def _default_trial_early_stop(report: TrialReport, reports: List[TrialReport], rungs: List['RungHeap']) -> bool:
    return False