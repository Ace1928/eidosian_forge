import gc
import tempfile
import unittest
import pytest
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
def test_value_head_not_str(self):
    """
        Test if the v-head is added to the model successfully, by passing a non `PretrainedModel`
        as an argument to `from_pretrained`.
        """
    for model_name in self.all_model_names:
        pretrained_model = self.transformers_model_class.from_pretrained(model_name)
        model = self.trl_model_class.from_pretrained(pretrained_model)
        assert hasattr(model, 'v_head')