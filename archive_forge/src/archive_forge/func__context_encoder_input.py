from typing import Any, Dict
import torch
from parlai.agents.image_seq2seq.modules import ContextWithImageEncoder
from parlai.agents.transformer.modules import get_n_positions_from_options
from parlai.agents.transformer.polyencoder import PolyencoderAgent, PolyEncoderModule
from parlai.core.torch_agent import Batch
from parlai.core.torch_image_agent import TorchImageAgent
from parlai.utils.misc import warn_once
def _context_encoder_input(self, ctxt_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
        Override PolyEncoderModule's inputs into the context encoder.
        """
    assert set(ctxt_inputs.keys()) == {'ctxt_tokens', 'ctxt_image'}
    return {'src_tokens': ctxt_inputs['ctxt_tokens'], 'image_features': ctxt_inputs['ctxt_image']}