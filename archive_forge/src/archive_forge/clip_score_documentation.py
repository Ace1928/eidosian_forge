from typing import TYPE_CHECKING, List, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _TRANSFORMERS_GREATER_EQUAL_4_10
Calculate `CLIP Score`_ which is a text-to-image similarity metric.

    CLIP Score is a reference free metric that can be used to evaluate the correlation between a generated caption for
    an image and the actual content of the image. It has been found to be highly correlated with human judgement. The
    metric is defined as:

    .. math::
        \text{CLIPScore(I, C)} = max(100 * cos(E_I, E_C), 0)

    which corresponds to the cosine similarity between visual `CLIP`_ embedding :math:`E_i` for an image :math:`i` and
    textual CLIP embedding :math:`E_C` for an caption :math:`C`. The score is bound between 0 and 100 and the closer
    to 100 the better.

    .. note:: Metric is not scriptable

    Args:
        images: Either a single [N, C, H, W] tensor or a list of [C, H, W] tensors
        text: Either a single caption or a list of captions
        model_name_or_path: string indicating the version of the CLIP model to use. Available models are
            `"openai/clip-vit-base-patch16"`, `"openai/clip-vit-base-patch32"`, `"openai/clip-vit-large-patch14-336"`
            and `"openai/clip-vit-large-patch14"`,

    Raises:
        ModuleNotFoundError:
            If transformers package is not installed or version is lower than 4.10.0
        ValueError:
            If not all images have format [C, H, W]
        ValueError:
            If the number of images and captions do not match

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> from torchmetrics.functional.multimodal import clip_score
        >>> score = clip_score(torch.randint(255, (3, 224, 224)), "a photo of a cat", "openai/clip-vit-base-patch16")
        >>> score.detach()
        tensor(24.4255)

    