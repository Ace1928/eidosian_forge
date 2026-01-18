import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import torch
from xformers._deprecation_warning import deprecated_function
from xformers.components import reversible as rv
from xformers.components.residual import ResidualNormStyle, get_deepnorm_coefficients
from xformers.factory.block_configs import (
from xformers.factory.block_factory import xFormerDecoderBlock, xFormerEncoderBlock
from xformers.factory.weight_init import get_weight_init_fn, xFormerWeightInit

        Given a serialized configuration, generate the corresponding model.
        This is only a helper and can easily be bypassed
        