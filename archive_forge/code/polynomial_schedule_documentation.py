from typing import Optional
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.schedules.schedule import Schedule
from ray.rllib.utils.typing import TensorType
Returns the result of:
        final_p + (initial_p - final_p) * (1 - `t`/t_max) ** power
        