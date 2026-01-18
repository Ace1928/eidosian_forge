import argparse
import json
import math
from typing import Tuple
import torch
from transformers import AutoTokenizer, MambaConfig, MambaForCausalLM
from transformers.utils import logging
from transformers.utils.import_utils import is_mamba_ssm_available
def validate_converted_model(original_state_dict: dict, original_ssm_config_dict: dict, hf_model: MambaForCausalLM, tokenizer: AutoTokenizer) -> None:
    """Validate the converted model returns the same output as the original model."""
    torch_device = 'cuda'
    original_config = MambaConfigSSM(**original_ssm_config_dict)
    original_model = MambaLMHeadModel(original_config).to(torch_device)
    original_model.load_state_dict(original_state_dict)
    hf_model = hf_model.to(torch_device)
    input_ids = tokenizer('Hey how are you doing?', return_tensors='pt')['input_ids'].to(torch_device)
    with torch.no_grad():
        original_model_logits = original_model(input_ids).logits
        hf_model_logits = hf_model(input_ids).logits
    if not torch.allclose(original_model_logits, hf_model_logits, atol=0.001):
        raise ValueError('The converted model did not return the same logits as the original model.')
    logger.info('Model conversion validated successfully.')