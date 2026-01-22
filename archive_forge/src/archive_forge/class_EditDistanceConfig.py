from typing import Any, Callable, Dict, Literal, Optional
from typing_extensions import TypedDict
class EditDistanceConfig(TypedDict, total=False):
    metric: METRICS
    normalize_score: bool