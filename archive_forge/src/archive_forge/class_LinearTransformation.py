import warnings
from typing import Any, Callable, cast, Dict, List, Optional, Sequence, Type, Union
import PIL.Image
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision import transforms as _transforms, tv_tensors
from torchvision.transforms.v2 import functional as F, Transform
from ._utils import _parse_labels_getter, _setup_number_or_seq, _setup_size, get_bounding_boxes, has_any, is_pure_tensor
class LinearTransformation(Transform):
    """[BETA] Transform a tensor image or video with a square transformation matrix and a mean_vector computed offline.

    .. v2betastatus:: LinearTransformation transform

    This transform does not support PIL Image.
    Given transformation_matrix and mean_vector, will flatten the torch.*Tensor and
    subtract mean_vector from it which is then followed by computing the dot
    product with the transformation matrix and then reshaping the tensor to its
    original shape.

    Applications:
        whitening transformation: Suppose X is a column vector zero-centered data.
        Then compute the data covariance matrix [D x D] with torch.mm(X.t(), X),
        perform SVD on this matrix and pass it as transformation_matrix.

    Args:
        transformation_matrix (Tensor): tensor [D x D], D = C x H x W
        mean_vector (Tensor): tensor [D], D = C x H x W
    """
    _v1_transform_cls = _transforms.LinearTransformation
    _transformed_types = (is_pure_tensor, tv_tensors.Image, tv_tensors.Video)

    def __init__(self, transformation_matrix: torch.Tensor, mean_vector: torch.Tensor):
        super().__init__()
        if transformation_matrix.size(0) != transformation_matrix.size(1):
            raise ValueError(f'transformation_matrix should be square. Got {tuple(transformation_matrix.size())} rectangular matrix.')
        if mean_vector.size(0) != transformation_matrix.size(0):
            raise ValueError(f'mean_vector should have the same length {mean_vector.size(0)} as any one of the dimensions of the transformation_matrix [{tuple(transformation_matrix.size())}]')
        if transformation_matrix.device != mean_vector.device:
            raise ValueError(f'Input tensors should be on the same device. Got {transformation_matrix.device} and {mean_vector.device}')
        if transformation_matrix.dtype != mean_vector.dtype:
            raise ValueError(f'Input tensors should have the same dtype. Got {transformation_matrix.dtype} and {mean_vector.dtype}')
        self.transformation_matrix = transformation_matrix
        self.mean_vector = mean_vector

    def _check_inputs(self, sample: Any) -> Any:
        if has_any(sample, PIL.Image.Image):
            raise TypeError(f'{type(self).__name__}() does not support PIL images.')

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        shape = inpt.shape
        n = shape[-3] * shape[-2] * shape[-1]
        if n != self.transformation_matrix.shape[0]:
            raise ValueError('Input tensor and transformation matrix have incompatible shape.' + f'[{shape[-3]} x {shape[-2]} x {shape[-1]}] != ' + f'{self.transformation_matrix.shape[0]}')
        if inpt.device.type != self.mean_vector.device.type:
            raise ValueError(f'Input tensor should be on the same device as transformation matrix and mean vector. Got {inpt.device} vs {self.mean_vector.device}')
        flat_inpt = inpt.reshape(-1, n) - self.mean_vector
        transformation_matrix = self.transformation_matrix.to(flat_inpt.dtype)
        output = torch.mm(flat_inpt, transformation_matrix)
        output = output.reshape(shape)
        if isinstance(inpt, (tv_tensors.Image, tv_tensors.Video)):
            output = tv_tensors.wrap(output, like=inpt)
        return output