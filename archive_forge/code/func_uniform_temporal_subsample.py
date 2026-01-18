import torch
from torchvision import tv_tensors
from torchvision.utils import _log_api_usage_once
from ._utils import _get_kernel, _register_kernel_internal
def uniform_temporal_subsample(inpt: torch.Tensor, num_samples: int) -> torch.Tensor:
    """[BETA] See :class:`~torchvision.transforms.v2.UniformTemporalSubsample` for details."""
    if torch.jit.is_scripting():
        return uniform_temporal_subsample_video(inpt, num_samples=num_samples)
    _log_api_usage_once(uniform_temporal_subsample)
    kernel = _get_kernel(uniform_temporal_subsample, type(inpt))
    return kernel(inpt, num_samples=num_samples)