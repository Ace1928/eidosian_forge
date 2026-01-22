from typing import List
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import TensorType, TensorStructType
def _unbatch_helper(v: TensorStructType, max_len: int) -> TensorStructType:
    """Recursively unpacks the repeat dimension (max_len)."""
    if isinstance(v, dict):
        return {k: _unbatch_helper(u, max_len) for k, u in v.items()}
    elif isinstance(v, tuple):
        return tuple((_unbatch_helper(u, max_len) for u in v))
    elif isinstance(v, RepeatedValues):
        unbatched = _unbatch_helper(v.values, max_len)
        return [RepeatedValues(u, v.lengths[:, i, ...], v.max_len) for i, u in enumerate(unbatched)]
    else:
        return [v[:, i, ...] for i in range(max_len)]