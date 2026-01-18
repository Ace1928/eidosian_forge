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
def token_classification(self, text: str, *, model: Optional[str]=None) -> List[TokenClassificationOutput]:
    """
        Perform token classification on the given text.
        Usually used for sentence parsing, either grammatical, or Named Entity Recognition (NER) to understand keywords contained within text.

        Args:
            text (`str`):
                A string to be classified.
            model (`str`, *optional*):
                The model to use for the token classification task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended token classification model will be used.
                Defaults to None.

        Returns:
            `List[Dict]`: List of token classification outputs containing the entity group, confidence score, word, start and end index.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.token_classification("My name is Sarah Jessica Parker but you can call me Jessica")
        [{'entity_group': 'PER',
        'score': 0.9971321225166321,
        'word': 'Sarah Jessica Parker',
        'start': 11,
        'end': 31},
        {'entity_group': 'PER',
        'score': 0.9773476123809814,
        'word': 'Jessica',
        'start': 52,
        'end': 59}]
        ```
        """
    payload: Dict[str, Any] = {'inputs': text}
    response = self.post(json=payload, model=model, task='token-classification')
    return _bytes_to_list(response)