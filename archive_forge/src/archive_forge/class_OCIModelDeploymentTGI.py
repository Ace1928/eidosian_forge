import logging
from typing import Any, Dict, List, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env
class OCIModelDeploymentTGI(OCIModelDeploymentLLM):
    """OCI Data Science Model Deployment TGI Endpoint.

    To use, you must provide the model HTTP endpoint from your deployed
    model, e.g. https://<MD_OCID>/predict.

    To authenticate, `oracle-ads` has been used to automatically load
    credentials: https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/authentication.html

    Make sure to have the required policies to access the OCI Data
    Science Model Deployment endpoint. See:
    https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-policies-auth.htm#model_dep_policies_auth__predict-endpoint

    Example:
        .. code-block:: python

            from langchain_community.llms import ModelDeploymentTGI

            oci_md = ModelDeploymentTGI(endpoint="https://<MD_OCID>/predict")

    """
    do_sample: bool = True
    'If set to True, this parameter enables decoding strategies such as\n    multi-nominal sampling, beam-search multi-nominal sampling, Top-K\n    sampling and Top-p sampling.\n    '
    watermark = True
    'Watermarking with `A Watermark for Large Language Models <https://arxiv.org/abs/2301.10226>`_.\n    Defaults to True.'
    return_full_text = False
    'Whether to prepend the prompt to the generated text. Defaults to False.'

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return 'oci_model_deployment_tgi_endpoint'

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for invoking OCI model deployment TGI endpoint."""
        return {'best_of': self.best_of, 'max_new_tokens': self.max_tokens, 'temperature': self.temperature, 'top_k': self.k if self.k > 0 else None, 'top_p': self.p, 'do_sample': self.do_sample, 'return_full_text': self.return_full_text, 'watermark': self.watermark}

    def _construct_json_body(self, prompt: str, params: dict) -> dict:
        return {'inputs': prompt, 'parameters': params}

    def _process_response(self, response_json: dict) -> str:
        return str(response_json.get('generated_text', response_json)) + '\n'