import enum
import inspect
from typing import Iterable, List, Optional, Tuple, Union
class BackboneConfigMixin:
    """
    A Mixin to support handling the `out_features` and `out_indices` attributes for the backbone configurations.
    """

    @property
    def out_features(self):
        return self._out_features

    @out_features.setter
    def out_features(self, out_features: List[str]):
        """
        Set the out_features attribute. This will also update the out_indices attribute to match the new out_features.
        """
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(out_features=out_features, out_indices=None, stage_names=self.stage_names)

    @property
    def out_indices(self):
        return self._out_indices

    @out_indices.setter
    def out_indices(self, out_indices: Union[Tuple[int], List[int]]):
        """
        Set the out_indices attribute. This will also update the out_features attribute to match the new out_indices.
        """
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(out_features=None, out_indices=out_indices, stage_names=self.stage_names)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default `to_dict()` from `PretrainedConfig` to
        include the `out_features` and `out_indices` attributes.
        """
        output = super().to_dict()
        output['out_features'] = output.pop('_out_features')
        output['out_indices'] = output.pop('_out_indices')
        return output