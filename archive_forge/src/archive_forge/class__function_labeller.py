from __future__ import annotations
import typing
from abc import ABCMeta, abstractmethod
from ..exceptions import PlotnineError
class _function_labeller(_core_labeller):
    """
    Use a function turn facet value into a label

    Parameters
    ----------
    func : callable
        Function to label an individual string
    """

    def __init__(self, func: Callable[[str], str]):
        self.func = func

    def __call__(self, label_info: strip_label_details) -> strip_label_details:
        label_info = label_info.copy()
        variables = label_info.variables
        for facet_var, facet_value in variables.items():
            variables[facet_var] = self.func(facet_value)
        return label_info