from typing import Any, Callable, Dict, Type
from langchain_core._api.deprecation import warn_deprecated
from langchain_core.language_models.llms import BaseLLM
def _import_oci_md_vllm() -> Type[BaseLLM]:
    from langchain_community.llms.oci_data_science_model_deployment_endpoint import OCIModelDeploymentVLLM
    return OCIModelDeploymentVLLM