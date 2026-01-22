import json
import logging
from abc import ABC
from copy import deepcopy
from typing import Any, Dict, List, Optional
from zlib import crc32
from ray._private.pydantic_compat import BaseModel
from ray.serve._private.config import DeploymentConfig
from ray.serve._private.utils import DeploymentOptionUpdateType, get_random_letters
from ray.serve.generated.serve_pb2 import DeploymentVersion as DeploymentVersionProto
def _get_serialized_options(self, update_types: List[DeploymentOptionUpdateType]) -> bytes:
    """Returns a serialized dictionary containing fields of a deployment config that
        should prompt a deployment version update.
        """
    reconfigure_dict = {}
    fields = self.deployment_config.model_fields if hasattr(self.deployment_config, 'model_fields') else self.deployment_config.__fields__
    for option_name, field in fields.items():
        option_weight = field.field_info.extra.get('update_type')
        if option_weight in update_types:
            reconfigure_dict[option_name] = getattr(self.deployment_config, option_name)
            if isinstance(reconfigure_dict[option_name], BaseModel):
                reconfigure_dict[option_name] = reconfigure_dict[option_name].dict()
    if isinstance(self.deployment_config.user_config, bytes) and 'user_config' in reconfigure_dict:
        del reconfigure_dict['user_config']
        return self.deployment_config.user_config + _serialize(reconfigure_dict)
    return _serialize(reconfigure_dict)