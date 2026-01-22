import datetime
import math
import typing as t
from wandb.util import (
def _params_obj_to_json_obj(params_obj: t.Any, artifact: t.Optional['Artifact']=None) -> t.Any:
    """Helper method."""
    if params_obj.__class__ == dict:
        return {key: _params_obj_to_json_obj(params_obj[key], artifact) for key in params_obj}
    elif params_obj.__class__ in [list, set, tuple, frozenset]:
        return [_params_obj_to_json_obj(item, artifact) for item in list(params_obj)]
    elif isinstance(params_obj, Type):
        return params_obj.to_json(artifact)
    else:
        return params_obj