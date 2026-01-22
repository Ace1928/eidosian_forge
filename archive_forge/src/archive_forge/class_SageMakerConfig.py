import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union
import yaml
from ...utils import ComputeEnvironment, DistributedType, SageMakerDistributedType
from ...utils.constants import SAGEMAKER_PYTHON_VERSION, SAGEMAKER_PYTORCH_VERSION, SAGEMAKER_TRANSFORMERS_VERSION
@dataclass
class SageMakerConfig(BaseConfig):
    ec2_instance_type: str
    iam_role_name: str
    image_uri: Optional[str] = None
    profile: Optional[str] = None
    region: str = 'us-east-1'
    num_machines: int = 1
    gpu_ids: str = 'all'
    base_job_name: str = f'accelerate-sagemaker-{num_machines}'
    pytorch_version: str = SAGEMAKER_PYTORCH_VERSION
    transformers_version: str = SAGEMAKER_TRANSFORMERS_VERSION
    py_version: str = SAGEMAKER_PYTHON_VERSION
    sagemaker_inputs_file: str = None
    sagemaker_metrics_file: str = None
    additional_args: dict = None
    dynamo_config: dict = None