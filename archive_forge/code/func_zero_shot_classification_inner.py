from __future__ import annotations
import base64
import math
import re
import warnings
import httpx
import yaml
from huggingface_hub import InferenceClient
from gradio import components
def zero_shot_classification_inner(input: str, labels: str, multi_label: bool):
    return client.zero_shot_classification(input, labels.split(','), multi_label=multi_label)