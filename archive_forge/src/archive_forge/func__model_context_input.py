from typing import Any, Dict
import torch
from parlai.agents.image_seq2seq.modules import ContextWithImageEncoder
from parlai.agents.transformer.modules import get_n_positions_from_options
from parlai.agents.transformer.polyencoder import PolyencoderAgent, PolyEncoderModule
from parlai.core.torch_agent import Batch
from parlai.core.torch_image_agent import TorchImageAgent
from parlai.utils.misc import warn_once
def _model_context_input(self, batch) -> Dict[str, Any]:
    """
        Override PolyencoderAgent's context inputs into the model.
        """
    return {'ctxt_tokens': batch.text_vec, 'ctxt_image': batch.image}