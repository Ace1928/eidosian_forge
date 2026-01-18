from typing import Union
from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging
from .base import Pipeline, build_pipeline_init_args

            Supports the following format
            - {"image": image, "question": question}
            - [{"image": image, "question": question}]
            - Generator and datasets
            