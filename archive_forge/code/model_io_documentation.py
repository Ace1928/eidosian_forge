import logging
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_STATE
from mlflow.transformers.flavor_config import FlavorKey, get_peft_base_model, is_peft_model

    Try to load a model with various loading strategies.
      1. Try to load the model with accelerate
      2. Try to load the model with the specified device
      3. Load the model without the device
    