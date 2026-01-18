import gc
import tempfile
import unittest
import pytest
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
def test_transformers_bf16_kwargs(self):
    """
        Test if the transformers kwargs are correctly passed
        Here we check that loading a model in half precision works as expected, i.e. the weights of
        the `pretrained_model` attribute is loaded in half precision and you can run a dummy
        forward pass without any issue.
        """
    for model_name in self.all_model_names:
        trl_model = self.trl_model_class.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        lm_head_namings = self.trl_model_class.lm_head_namings
        if model_name == 'trl-internal-testing/tiny-random-FSMTForConditionalGeneration':
            continue
        assert any((hasattr(trl_model.pretrained_model, lm_head_naming) for lm_head_naming in lm_head_namings))
        for lm_head_naming in lm_head_namings:
            if hasattr(trl_model.pretrained_model, lm_head_naming):
                assert getattr(trl_model.pretrained_model, lm_head_naming).weight.dtype == torch.bfloat16
        dummy_input = torch.LongTensor([[0, 1, 0, 1]])
        _ = trl_model(input_ids=dummy_input, decoder_input_ids=dummy_input)