import base64
import logging
import time
import warnings
from dataclasses import asdict
from typing import (
from requests import HTTPError
from requests.structures import CaseInsensitiveDict
from huggingface_hub.constants import ALL_INFERENCE_API_FRAMEWORKS, INFERENCE_ENDPOINT, MAIN_INFERENCE_API_FRAMEWORKS
from huggingface_hub.inference._common import (
from huggingface_hub.inference._text_generation import (
from huggingface_hub.inference._types import (
from huggingface_hub.utils import (
def visual_question_answering(self, image: ContentT, question: str, *, model: Optional[str]=None) -> List[str]:
    """
        Answering open-ended questions based on an image.

        Args:
            image (`Union[str, Path, bytes, BinaryIO]`):
                The input image for the context. It can be raw bytes, an image file, or a URL to an online image.
            question (`str`):
                Question to be answered.
            model (`str`, *optional*):
                The model to use for the visual question answering task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended visual question answering model will be used.
                Defaults to None.

        Returns:
            `List[Dict]`: a list of dictionaries containing the predicted label and associated probability.

        Raises:
            `InferenceTimeoutError`:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.visual_question_answering(
        ...     image="https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg",
        ...     question="What is the animal doing?"
        ... )
        [{'score': 0.778609573841095, 'answer': 'laying down'},{'score': 0.6957435607910156, 'answer': 'sitting'}, ...]
        ```
        """
    payload: Dict[str, Any] = {'question': question, 'image': _b64_encode(image)}
    response = self.post(json=payload, model=model, task='visual-question-answering')
    return _bytes_to_list(response)