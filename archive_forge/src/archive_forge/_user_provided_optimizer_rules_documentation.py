from typing import List
from ray.data._internal.logical.interfaces.optimizer import Rule

    Users can provide extra physical optimization rules here
    to be used in `PhysicalOptimizer`.

    Args:
        default_rules: the default physical optimization rules.

    Returns:
        The final physical optimization rules to be used in `PhysicalOptimizer`.
    