from typing import Dict
from ..utils import add_end_docstrings
from .base import GenericTensor, Pipeline, build_pipeline_init_args

        Extract the features of the input(s).

        Args:
            args (`str` or `List[str]`): One or several texts (or one list of texts) to get the features of.

        Return:
            A nested list of `float`: The features computed by the model.
        