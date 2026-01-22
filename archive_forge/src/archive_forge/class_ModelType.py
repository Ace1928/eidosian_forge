import re
from typing import List, Optional, Tuple, Union
import numpy as np
from ..utils import (
from .base import ChunkPipeline, build_pipeline_init_args
from .question_answering import select_starts_ends
class ModelType(ExplicitEnum):
    LayoutLM = 'layoutlm'
    LayoutLMv2andv3 = 'layoutlmv2andv3'
    VisionEncoderDecoder = 'vision_encoder_decoder'