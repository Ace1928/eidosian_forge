import gc
import tempfile
import unittest
import pytest
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
def test_raise_error_not_causallm(self):
    model_id = 'trl-internal-testing/tiny-random-T5Model'
    with pytest.raises(ValueError):
        pretrained_model = AutoModel.from_pretrained(model_id)
        _ = self.trl_model_class.from_pretrained(pretrained_model)