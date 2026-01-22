import warnings
from torch import Tensor
from .batchnorm import _LazyNormBase, _NormBase
from .. import functional as F
class _InstanceNorm(_NormBase):

    def __init__(self, num_features: int, eps: float=1e-05, momentum: float=0.1, affine: bool=False, track_running_stats: bool=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(num_features, eps, momentum, affine, track_running_stats, **factory_kwargs)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def _get_no_batch_dim(self):
        raise NotImplementedError

    def _handle_no_batch_input(self, input):
        return self._apply_instance_norm(input.unsqueeze(0)).squeeze(0)

    def _apply_instance_norm(self, input):
        return F.instance_norm(input, self.running_mean, self.running_var, self.weight, self.bias, self.training or not self.track_running_stats, self.momentum, self.eps)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None and (not self.track_running_stats):
            running_stats_keys = []
            for name in ('running_mean', 'running_var'):
                key = prefix + name
                if key in state_dict:
                    running_stats_keys.append(key)
            if len(running_stats_keys) > 0:
                error_msgs.append('Unexpected running stats buffer(s) {names} for {klass} with track_running_stats=False. If state_dict is a checkpoint saved before 0.4.0, this may be expected because {klass} does not track running stats by default since 0.4.0. Please remove these keys from state_dict. If the running stats are actually needed, instead set track_running_stats=True in {klass} to enable them. See the documentation of {klass} for details.'.format(names=' and '.join((f'"{k}"' for k in running_stats_keys)), klass=self.__class__.__name__))
                for key in running_stats_keys:
                    state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        feature_dim = input.dim() - self._get_no_batch_dim()
        if input.size(feature_dim) != self.num_features:
            if self.affine:
                raise ValueError(f"expected input's size at dim={feature_dim} to match num_features ({self.num_features}), but got: {input.size(feature_dim)}.")
            else:
                warnings.warn(f"input's size at dim={feature_dim} does not match num_features. You can silence this warning by not passing in num_features, which is not used because affine=False")
        if input.dim() == self._get_no_batch_dim():
            return self._handle_no_batch_input(input)
        return self._apply_instance_norm(input)