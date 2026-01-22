from typing import TYPE_CHECKING, Dict, List, Literal, Tuple, Union
import torch
from torch import Tensor
from torchmetrics.functional.multimodal.clip_score import _get_clip_model_and_processor
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _PIQ_GREATER_EQUAL_0_8, _TRANSFORMERS_GREATER_EQUAL_4_10
Calculates `CLIP-IQA`_, that can be used to measure the visual content of images.

    The metric is based on the `CLIP`_ model, which is a neural network trained on a variety of (image, text) pairs to
    be able to generate a vector representation of the image and the text that is similar if the image and text are
    semantically similar.

    The metric works by calculating the cosine similarity between user provided images and pre-defined prompts. The
    prompts always come in pairs of "positive" and "negative" such as "Good photo." and "Bad photo.". By calculating
    the similartity between image embeddings and both the "positive" and "negative" prompt, the metric can determine
    which prompt the image is more similar to. The metric then returns the probability that the image is more similar
    to the first prompt than the second prompt.

    Build in prompts are:
        * quality: "Good photo." vs "Bad photo."
        * brightness: "Bright photo." vs "Dark photo."
        * noisiness: "Clean photo." vs "Noisy photo."
        * colorfullness: "Colorful photo." vs "Dull photo."
        * sharpness: "Sharp photo." vs "Blurry photo."
        * contrast: "High contrast photo." vs "Low contrast photo."
        * complexity: "Complex photo." vs "Simple photo."
        * natural: "Natural photo." vs "Synthetic photo."
        * happy: "Happy photo." vs "Sad photo."
        * scary: "Scary photo." vs "Peaceful photo."
        * new: "New photo." vs "Old photo."
        * warm: "Warm photo." vs "Cold photo."
        * real: "Real photo." vs "Abstract photo."
        * beautiful: "Beautiful photo." vs "Ugly photo."
        * lonely: "Lonely photo." vs "Sociable photo."
        * relaxing: "Relaxing photo." vs "Stressful photo."

    Args:
        images: Either a single ``[N, C, H, W]`` tensor or a list of ``[C, H, W]`` tensors
        model_name_or_path: string indicating the version of the CLIP model to use. By default this argument is set to
            ``clip_iqa`` which corresponds to the model used in the original paper. Other available models are
            `"openai/clip-vit-base-patch16"`, `"openai/clip-vit-base-patch32"`, `"openai/clip-vit-large-patch14-336"`
            and `"openai/clip-vit-large-patch14"`
        data_range: The maximum value of the input tensor. For example, if the input images are in range [0, 255],
            data_range should be 255. The images are normalized by this value.
        prompts: A string, tuple of strings or nested tuple of strings. If a single string is provided, it must be one
            of the available prompts (see above). Else the input is expected to be a tuple, where each element can
            be one of two things: either a string or a tuple of strings. If a string is provided, it must be one of the
            available prompts (see above). If tuple is provided, it must be of length 2 and the first string must be a
            positive prompt and the second string must be a negative prompt.

    .. note:: If using the default `clip_iqa` model, the package `piq` must be installed. Either install with
        `pip install piq` or `pip install torchmetrics[multimodal]`.

    Returns:
        A tensor of shape ``(N,)`` if a single prompts is provided. If a list of prompts is provided, a dictionary of
        with the prompts as keys and tensors of shape ``(N,)`` as values.

    Raises:
        ModuleNotFoundError:
            If transformers package is not installed or version is lower than 4.10.0
        ValueError:
            If not all images have format [C, H, W]
        ValueError:
            If prompts is a tuple and it is not of length 2
        ValueError:
            If prompts is a string and it is not one of the available prompts
        ValueError:
            If prompts is a list of strings and not all strings are one of the available prompts

    Example::
        Single prompt:

        >>> from torchmetrics.functional.multimodal import clip_image_quality_assessment
        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> imgs = torch.randint(255, (2, 3, 224, 224)).float()
        >>> clip_image_quality_assessment(imgs, prompts=("quality",))
        tensor([0.8894, 0.8902])

    Example::
        Multiple prompts:

        >>> from torchmetrics.functional.multimodal import clip_image_quality_assessment
        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> imgs = torch.randint(255, (2, 3, 224, 224)).float()
        >>> clip_image_quality_assessment(imgs, prompts=("quality", "brightness"))
        {'quality': tensor([0.8894, 0.8902]), 'brightness': tensor([0.5507, 0.5208])}

    Example::
        Custom prompts. Must always be a tuple of length 2, with a positive and negative prompt.

        >>> from torchmetrics.functional.multimodal import clip_image_quality_assessment
        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> imgs = torch.randint(255, (2, 3, 224, 224)).float()
        >>> clip_image_quality_assessment(imgs, prompts=(("Super good photo.", "Super bad photo."), "brightness"))
        {'user_defined_0': tensor([0.9652, 0.9629]), 'brightness': tensor([0.5507, 0.5208])}

    