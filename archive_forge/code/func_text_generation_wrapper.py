from __future__ import annotations
import base64
import math
import re
import warnings
import httpx
import yaml
from huggingface_hub import InferenceClient
from gradio import components
def text_generation_wrapper(client: InferenceClient):

    def text_generation_inner(input: str):
        return input + client.text_generation(input)
    return text_generation_inner