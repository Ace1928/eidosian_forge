from typing import Sequence, Any, Dict, Tuple, Callable, Optional, TypeVar, Union
from typing import List
import inspect
def _get_entry_points(self) -> List[importlib_metadata.EntryPoint]:
    if hasattr(AVAILABLE_ENTRY_POINTS, 'select'):
        return AVAILABLE_ENTRY_POINTS.select(group=self.entry_point_namespace)
    else:
        return AVAILABLE_ENTRY_POINTS.get(self.entry_point_namespace, [])