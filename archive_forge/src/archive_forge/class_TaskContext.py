from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional
from ray.data._internal.progress_bar import ProgressBar
@dataclass
class TaskContext:
    """This describes the information of a task running block transform."""
    task_idx: int
    sub_progress_bar_dict: Optional[Dict[str, ProgressBar]] = None
    upstream_map_transformer: Optional['MapTransformer'] = None
    upstream_map_ray_remote_args: Optional[Dict[str, Any]] = None
    target_max_block_size: Optional[int] = None