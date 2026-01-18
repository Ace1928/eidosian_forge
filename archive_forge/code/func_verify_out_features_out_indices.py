import enum
import inspect
from typing import Iterable, List, Optional, Tuple, Union
def verify_out_features_out_indices(out_features: Optional[Iterable[str]], out_indices: Optional[Iterable[int]], stage_names: Optional[Iterable[str]]):
    """
    Verify that out_indices and out_features are valid for the given stage_names.
    """
    if stage_names is None:
        raise ValueError('Stage_names must be set for transformers backbones')
    if out_features is not None:
        if not isinstance(out_features, (list,)):
            raise ValueError(f'out_features must be a list got {type(out_features)}')
        if any((feat not in stage_names for feat in out_features)):
            raise ValueError(f'out_features must be a subset of stage_names: {stage_names} got {out_features}')
        if len(out_features) != len(set(out_features)):
            raise ValueError(f'out_features must not contain any duplicates, got {out_features}')
        if out_features != (sorted_feats := [feat for feat in stage_names if feat in out_features]):
            raise ValueError(f'out_features must be in the same order as stage_names, expected {sorted_feats} got {out_features}')
    if out_indices is not None:
        if not isinstance(out_indices, (list, tuple)):
            raise ValueError(f'out_indices must be a list or tuple, got {type(out_indices)}')
        positive_indices = tuple((idx % len(stage_names) if idx < 0 else idx for idx in out_indices))
        if any((idx for idx in positive_indices if idx not in range(len(stage_names)))):
            raise ValueError(f'out_indices must be valid indices for stage_names {stage_names}, got {out_indices}')
        if len(positive_indices) != len(set(positive_indices)):
            msg = f'out_indices must not contain any duplicates, got {out_indices}'
            msg += f'(equivalent to {positive_indices}))' if positive_indices != out_indices else ''
            raise ValueError(msg)
        if positive_indices != tuple(sorted(positive_indices)):
            sorted_negative = tuple((idx for _, idx in sorted(zip(positive_indices, out_indices), key=lambda x: x[0])))
            raise ValueError(f'out_indices must be in the same order as stage_names, expected {sorted_negative} got {out_indices}')
    if out_features is not None and out_indices is not None:
        if len(out_features) != len(out_indices):
            raise ValueError('out_features and out_indices should have the same length if both are set')
        if out_features != [stage_names[idx] for idx in out_indices]:
            raise ValueError('out_features and out_indices should correspond to the same stages if both are set')