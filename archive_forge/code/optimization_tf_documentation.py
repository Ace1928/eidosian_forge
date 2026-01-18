import re
from typing import Callable, List, Optional, Union
import tensorflow as tf
from .modeling_tf_utils import keras
Resets the accumulated gradients on the current replica.