import math
from typing import List, Optional, Tuple
import torch
def _process_attention_output(self, rc_output: torch.Tensor, utterance: torch.Tensor, right_context: torch.Tensor) -> torch.Tensor:
    result = self.dropout(rc_output) + torch.cat([right_context, utterance])
    result = self.pos_ff(result) + result
    result = self.layer_norm_output(result)
    return result