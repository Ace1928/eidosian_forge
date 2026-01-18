from __future__ import annotations
import base64
import math
import re
import warnings
import httpx
import yaml
from huggingface_hub import InferenceClient
from gradio import components
def token_classification_wrapper(client: InferenceClient):

    def token_classification_inner(input: str):
        ner_list = client.token_classification(input)
        return format_ner_list(input, ner_list)
    return token_classification_inner