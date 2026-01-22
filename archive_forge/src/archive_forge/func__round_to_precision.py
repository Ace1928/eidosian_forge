from __future__ import annotations
from typing import Any, Callable
from gradio_client.documentation import document
from gradio.components.base import FormComponent
from gradio.events import Events
from gradio.exceptions import Error
@staticmethod
def _round_to_precision(num: float | int, precision: int | None) -> float | int:
    """
        Round to a given precision.

        If precision is None, no rounding happens. If 0, num is converted to int.

        Parameters:
            num: Number to round.
            precision: Precision to round to.
        Returns:
            rounded number or the original number if precision is None
        """
    if precision is None:
        return num
    elif precision == 0:
        return int(round(num, precision))
    else:
        return round(num, precision)