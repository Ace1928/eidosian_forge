import enum
import inspect
from typing import Iterable, List, Optional, Tuple, Union
class BackboneMixin:
    backbone_type: Optional[BackboneType] = None

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

    def _init_transformers_backbone(self, config) -> None:
        stage_names = getattr(config, 'stage_names')
        out_features = getattr(config, 'out_features', None)
        out_indices = getattr(config, 'out_indices', None)
        self.stage_names = stage_names
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(out_features=out_features, out_indices=out_indices, stage_names=stage_names)
        self.num_features = None

    def _init_backbone(self, config) -> None:
        """
        Method to initialize the backbone. This method is called by the constructor of the base class after the
        pretrained model weights have been loaded.
        """
        self.config = config
        self.use_timm_backbone = getattr(config, 'use_timm_backbone', False)
        self.backbone_type = BackboneType.TIMM if self.use_timm_backbone else BackboneType.TRANSFORMERS
        if self.backbone_type == BackboneType.TIMM:
            self._init_timm_backbone(config)
        elif self.backbone_type == BackboneType.TRANSFORMERS:
            self._init_transformers_backbone(config)
        else:
            raise ValueError(f'backbone_type {self.backbone_type} not supported.')

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

    @property
    def out_feature_channels(self):
        return {stage: self.num_features[i] for i, stage in enumerate(self.stage_names)}

    @property
    def channels(self):
        return [self.out_feature_channels[name] for name in self.out_features]

    def forward_with_filtered_kwargs(self, *args, **kwargs):
        signature = dict(inspect.signature(self.forward).parameters)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in signature}
        return self(*args, **filtered_kwargs)

    def forward(self, pixel_values, output_hidden_states: Optional[bool]=None, output_attentions: Optional[bool]=None, return_dict: Optional[bool]=None):
        raise NotImplementedError('This method should be implemented by the derived class.')

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default `to_dict()` from `PretrainedConfig` to
        include the `out_features` and `out_indices` attributes.
        """
        output = super().to_dict()
        output['out_features'] = output.pop('_out_features')
        output['out_indices'] = output.pop('_out_indices')
        return output