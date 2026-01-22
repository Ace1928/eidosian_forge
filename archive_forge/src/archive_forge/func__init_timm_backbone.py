import enum
import inspect
from typing import Iterable, List, Optional, Tuple, Union
def _init_timm_backbone(self, config) -> None:
    """
        Initialize the backbone model from timm The backbone must already be loaded to self._backbone
        """
    if getattr(self, '_backbone', None) is None:
        raise ValueError('self._backbone must be set before calling _init_timm_backbone')
    self.stage_names = [stage['module'] for stage in self._backbone.feature_info.info]
    self.num_features = [stage['num_chs'] for stage in self._backbone.feature_info.info]
    out_indices = self._backbone.feature_info.out_indices
    out_features = self._backbone.feature_info.module_name()
    verify_out_features_out_indices(out_features=out_features, out_indices=out_indices, stage_names=self.stage_names)
    self._out_features, self._out_indices = (out_features, out_indices)