import torch
from torchvision import tv_tensors
from torchvision.utils import _log_api_usage_once
from ._utils import _get_kernel, _register_kernel_internal
@_register_kernel_internal(uniform_temporal_subsample, torch.Tensor)
@_register_kernel_internal(uniform_temporal_subsample, tv_tensors.Video)
def uniform_temporal_subsample_video(video: torch.Tensor, num_samples: int) -> torch.Tensor:
    t_max = video.shape[-4] - 1
    indices = torch.linspace(0, t_max, num_samples, device=video.device).long()
    return torch.index_select(video, -4, indices)