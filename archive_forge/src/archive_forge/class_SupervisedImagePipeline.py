from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
import tensorflow as tf
from tensorflow import keras
from autokeras import auto_model
from autokeras import blocks
from autokeras import nodes as input_module
from autokeras.engine import tuner
from autokeras.tuners import greedy
from autokeras.tuners import task_specific
from autokeras.utils import types
class SupervisedImagePipeline(auto_model.AutoModel):

    def __init__(self, outputs, **kwargs):
        super().__init__(inputs=input_module.ImageInput(), outputs=outputs, **kwargs)