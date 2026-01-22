import importlib
import logging
from typing import Any, Callable, List, Optional
from langchain_community.embeddings.self_hosted import SelfHostedEmbeddings
class SelfHostedHuggingFaceEmbeddings(SelfHostedEmbeddings):
    """HuggingFace embedding models on self-hosted remote hardware.

    Supported hardware includes auto-launched instances on AWS, GCP, Azure,
    and Lambda, as well as servers specified
    by IP address and SSH credentials (such as on-prem, or another cloud
    like Paperspace, Coreweave, etc.).

    To use, you should have the ``runhouse`` python package installed.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import SelfHostedHuggingFaceEmbeddings
            import runhouse as rh
            model_id = "sentence-transformers/all-mpnet-base-v2"
            gpu = rh.cluster(name="rh-a10x", instance_type="A100:1")
            hf = SelfHostedHuggingFaceEmbeddings(model_id=model_id, hardware=gpu)
    """
    client: Any
    model_id: str = DEFAULT_MODEL_NAME
    'Model name to use.'
    model_reqs: List[str] = ['./', 'sentence_transformers', 'torch']
    'Requirements to install on hardware to inference the model.'
    hardware: Any
    'Remote hardware to send the inference function to.'
    model_load_fn: Callable = load_embedding_model
    'Function to load the model remotely on the server.'
    load_fn_kwargs: Optional[dict] = None
    'Keyword arguments to pass to the model load function.'
    inference_fn: Callable = _embed_documents
    'Inference function to extract the embeddings.'

    def __init__(self, **kwargs: Any):
        """Initialize the remote inference function."""
        load_fn_kwargs = kwargs.pop('load_fn_kwargs', {})
        load_fn_kwargs['model_id'] = load_fn_kwargs.get('model_id', DEFAULT_MODEL_NAME)
        load_fn_kwargs['instruct'] = load_fn_kwargs.get('instruct', False)
        load_fn_kwargs['device'] = load_fn_kwargs.get('device', 0)
        super().__init__(load_fn_kwargs=load_fn_kwargs, **kwargs)