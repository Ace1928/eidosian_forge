from typing import Dict, List, Tuple
import torch
from .modules import ImageSeq2seqModel, FusionType
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.dict import DictionaryAgent
from parlai.core.torch_agent import Batch
from parlai.core.torch_image_agent import TorchImageAgent

        Override for custom loading.

        Reasons:
            1. When using an init model without an image encoder
            2. We decide to add segment embeddings after the fact.
            3. When using an init model with only an encoder provided
                In this case, we may need to add the START token to the state_dict
            4. When using an init model without image tokens in the embeddings.
                This is only the case if the embs differ by 2 in dimension 0
        