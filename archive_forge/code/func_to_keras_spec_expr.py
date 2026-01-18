from typing import Any, Type, Dict
from triad.utils.convert import get_full_type_path, to_type
from tune.concepts.space.parameters import TuningParametersTemplate
from tune_tensorflow.spec import KerasTrainingSpec
from tune import Space
from tune.constants import SPACE_MODEL_NAME
def to_keras_spec_expr(spec: Any) -> str:
    if isinstance(spec, str):
        spec = to_keras_spec(spec)
    return get_full_type_path(spec)