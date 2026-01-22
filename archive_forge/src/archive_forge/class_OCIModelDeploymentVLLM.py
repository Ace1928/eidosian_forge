import logging
from typing import Any, Dict, List, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env
class OCIModelDeploymentVLLM(OCIModelDeploymentLLM):
    """VLLM deployed on OCI Data Science Model Deployment

    To use, you must provide the model HTTP endpoint from your deployed
    model, e.g. https://<MD_OCID>/predict.

    To authenticate, `oracle-ads` has been used to automatically load
    credentials: https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/authentication.html

    Make sure to have the required policies to access the OCI Data
    Science Model Deployment endpoint. See:
    https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-policies-auth.htm#model_dep_policies_auth__predict-endpoint

    Example:
        .. code-block:: python

            from langchain_community.llms import OCIModelDeploymentVLLM

            oci_md = OCIModelDeploymentVLLM(
                endpoint="https://<MD_OCID>/predict",
                model="mymodel"
            )

    """
    model: str
    'The name of the model.'
    n: int = 1
    'Number of output sequences to return for the given prompt.'
    k: int = -1
    'Number of most likely tokens to consider at each step.'
    frequency_penalty: float = 0.0
    'Penalizes repeated tokens according to frequency. Between 0 and 1.'
    presence_penalty: float = 0.0
    'Penalizes repeated tokens. Between 0 and 1.'
    use_beam_search: bool = False
    'Whether to use beam search instead of sampling.'
    ignore_eos: bool = False
    'Whether to ignore the EOS token and continue generating tokens after\n    the EOS token is generated.'
    logprobs: Optional[int] = None
    'Number of log probabilities to return per output token.'

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return 'oci_model_deployment_vllm_endpoint'

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling vllm."""
        return {'best_of': self.best_of, 'frequency_penalty': self.frequency_penalty, 'ignore_eos': self.ignore_eos, 'logprobs': self.logprobs, 'max_tokens': self.max_tokens, 'model': self.model, 'n': self.n, 'presence_penalty': self.presence_penalty, 'stop': self.stop, 'temperature': self.temperature, 'top_k': self.k, 'top_p': self.p, 'use_beam_search': self.use_beam_search}

    def _construct_json_body(self, prompt: str, params: dict) -> dict:
        return {'prompt': prompt, **params}

    def _process_response(self, response_json: dict) -> str:
        return response_json['choices'][0]['text']