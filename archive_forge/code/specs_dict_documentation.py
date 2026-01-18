from typing import Union, Mapping, Any
from ray.rllib.utils.annotations import ExperimentalAPI, override
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.core.models.specs.specs_base import Spec
Checks whether the data matches the spec.

        Args:
            data: The data which should match the spec. It can also be a spec.
            exact_match: If true, the data and the spec must be exactly identical.
                Otherwise, the data is validated as long as it contains at least the
                elements of the spec, but can contain more entries.
        Raises:
            ValueError: If the data doesn't match the spec.
        