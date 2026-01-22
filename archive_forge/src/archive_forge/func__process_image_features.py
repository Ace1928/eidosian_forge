from typing import Dict, List, Tuple
import torch
from .modules import ImageSeq2seqModel, FusionType
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.dict import DictionaryAgent
from parlai.core.torch_agent import Batch
from parlai.core.torch_image_agent import TorchImageAgent
def _process_image_features(self, features: torch.Tensor) -> torch.Tensor:
    """
        Format shape and type of input image-feature tensor.

        Override TorchImageAgent._process_image_features to handle multi-dimensional
        images.
        """
    features = features.view(-1, self.image_features_dim)
    return torch.stack([TorchImageAgent._process_image_features(self, features[i]) for i in range(features.size(0))])