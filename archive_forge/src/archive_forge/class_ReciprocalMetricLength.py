from __future__ import annotations
import logging # isort:skip
from abc import abstractmethod
from typing import Any
from ...core.has_props import abstract
from ...core.properties import (
from ...model import Model
class ReciprocalMetricLength(ReciprocalMetric):
    """ Metric units of reciprocal length measurement.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
    base_unit = Override(default='m')
    exclude = Override(default=['dm', 'hm'])