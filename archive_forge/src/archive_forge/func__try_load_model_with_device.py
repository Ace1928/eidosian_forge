import logging
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_STATE
from mlflow.transformers.flavor_config import FlavorKey, get_peft_base_model, is_peft_model
def _try_load_model_with_device(model_class, model_name_or_path, load_kwargs):
    try:
        return model_class.from_pretrained(model_name_or_path, **load_kwargs)
    except OSError as e:
        revision = load_kwargs.get('revision')
        if f'{revision} is not a valid git identifier' in str(e):
            raise MlflowException(f"The model was saved with a HuggingFace Hub repository name '{model_name_or_path}'and a commit hash '{revision}', but the commit is not found in the repository. ")
        else:
            raise e
    except (ValueError, TypeError, NotImplementedError):
        pass