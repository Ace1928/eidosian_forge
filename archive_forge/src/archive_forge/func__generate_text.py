import importlib.util
import logging
import pickle
from typing import Any, Callable, List, Mapping, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra
from langchain_community.llms.utils import enforce_stop_tokens
def _generate_text(pipeline: Any, prompt: str, *args: Any, stop: Optional[List[str]]=None, **kwargs: Any) -> str:
    """Inference function to send to the remote hardware.

    Accepts a pipeline callable (or, more likely,
    a key pointing to the model on the cluster's object store)
    and returns text predictions for each document
    in the batch.
    """
    text = pipeline(prompt, *args, **kwargs)
    if stop is not None:
        text = enforce_stop_tokens(text, stop)
    return text