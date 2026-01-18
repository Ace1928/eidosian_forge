import gc
import tempfile
import unittest
import pytest
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
def test_value_head_init_random(self):
    """
        Test if the v-head has been randomly initialized.
        We can check that by making sure the bias is different
        than zeros by default.
        """
    for model_name in self.all_model_names:
        model = self.trl_model_class.from_pretrained(model_name)
        assert not torch.allclose(model.v_head.summary.bias, torch.zeros_like(model.v_head.summary.bias))