from typing import Optional
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.logical.interfaces import PhysicalPlan, Rule
For each op that has overridden the default target max block size,
    propagate to upstream ops until we reach an op that has also overridden the
    target max block size.