import gc
import tempfile
import unittest
import pytest
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
class VHeadModelTester:
    all_model_names = None
    trl_model_class = None
    transformers_model_class = None

    def test_value_head(self):
        """
        Test if the v-head is added to the model successfully
        """
        for model_name in self.all_model_names:
            model = self.trl_model_class.from_pretrained(model_name)
            assert hasattr(model, 'v_head')

    def test_value_head_shape(self):
        """
        Test if the v-head has the correct shape
        """
        for model_name in self.all_model_names:
            model = self.trl_model_class.from_pretrained(model_name)
            assert model.v_head.summary.weight.shape[0] == 1

    def test_value_head_init_random(self):
        """
        Test if the v-head has been randomly initialized.
        We can check that by making sure the bias is different
        than zeros by default.
        """
        for model_name in self.all_model_names:
            model = self.trl_model_class.from_pretrained(model_name)
            assert not torch.allclose(model.v_head.summary.bias, torch.zeros_like(model.v_head.summary.bias))

    def test_value_head_not_str(self):
        """
        Test if the v-head is added to the model successfully, by passing a non `PretrainedModel`
        as an argument to `from_pretrained`.
        """
        for model_name in self.all_model_names:
            pretrained_model = self.transformers_model_class.from_pretrained(model_name)
            model = self.trl_model_class.from_pretrained(pretrained_model)
            assert hasattr(model, 'v_head')

    def test_from_save_trl(self):
        """
        Test if the model can be saved and loaded from a directory and get the same weights
        Including the additional modules (e.g. v_head)
        """
        for model_name in self.all_model_names:
            model = self.trl_model_class.from_pretrained(model_name)
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.save_pretrained(tmp_dir)
                model_from_save = self.trl_model_class.from_pretrained(tmp_dir)
            for key in model_from_save.state_dict():
                assert torch.allclose(model_from_save.state_dict()[key], model.state_dict()[key])

    def test_from_save_trl_sharded(self):
        """
        Test if the model can be saved and loaded from a directory and get the same weights - sharded case
        """
        for model_name in self.all_model_names:
            model = self.trl_model_class.from_pretrained(model_name)
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.save_pretrained(tmp_dir)
                model_from_save = self.trl_model_class.from_pretrained(tmp_dir)
            for key in model_from_save.state_dict():
                assert torch.allclose(model_from_save.state_dict()[key], model.state_dict()[key])

    def test_from_save_transformers_sharded(self):
        """
        Test if the model can be saved and loaded using transformers and get the same weights - sharded case
        """
        for model_name in self.all_model_names:
            transformers_model = self.trl_model_class.transformers_parent_class.from_pretrained(model_name)
            trl_model = self.trl_model_class.from_pretrained(model_name)
            with tempfile.TemporaryDirectory() as tmp_dir:
                trl_model.save_pretrained(tmp_dir, max_shard_size='1MB')
                transformers_model_from_save = self.trl_model_class.transformers_parent_class.from_pretrained(tmp_dir)
            for key in transformers_model.state_dict():
                assert torch.allclose(transformers_model_from_save.state_dict()[key], transformers_model.state_dict()[key])

    def test_from_save_transformers(self):
        """
        Test if the model can be saved and loaded using transformers and get the same weights.
        We override the test of the super class to check if the weights are the same.
        """
        for model_name in self.all_model_names:
            transformers_model = self.trl_model_class.transformers_parent_class.from_pretrained(model_name)
            trl_model = self.trl_model_class.from_pretrained(model_name)
            with tempfile.TemporaryDirectory() as tmp_dir:
                trl_model.save_pretrained(tmp_dir)
                transformers_model_from_save = self.trl_model_class.transformers_parent_class.from_pretrained(tmp_dir)
            for key in transformers_model.state_dict():
                assert torch.allclose(transformers_model_from_save.state_dict()[key], transformers_model.state_dict()[key])
            for key in trl_model.state_dict():
                if 'v_head' not in key:
                    assert key in transformers_model.state_dict()
                    assert torch.allclose(trl_model.state_dict()[key], transformers_model.state_dict()[key])
            assert set(transformers_model_from_save.state_dict().keys()) == set(transformers_model.state_dict().keys())