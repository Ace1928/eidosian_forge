import gc
import tempfile
import unittest
import pytest
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
class Seq2SeqValueHeadModelTester(VHeadModelTester, unittest.TestCase):
    """
    Testing suite for v-head models.
    """
    all_model_names = ALL_SEQ2SEQ_MODELS
    trl_model_class = AutoModelForSeq2SeqLMWithValueHead
    transformers_model_class = AutoModelForSeq2SeqLM

    def tearDown(self):
        gc.collect()

    def test_inference(self):
        """
        Test if the model can be used for inference and outputs 3 values
        - logits, loss, and value states
        """
        EXPECTED_OUTPUT_SIZE = 3
        for model_name in self.all_model_names:
            model = self.trl_model_class.from_pretrained(model_name)
            input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            decoder_input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            outputs = model(input_ids, decoder_input_ids=decoder_input_ids)
            assert len(outputs) == EXPECTED_OUTPUT_SIZE

    def test_dropout_config(self):
        """
        Test if we instantiate a model by adding `summary_drop_prob` to the config
        it will be added to the v_head
        """
        for model_name in self.all_model_names:
            pretrained_model = self.transformers_model_class.from_pretrained(model_name)
            pretrained_model.config.summary_dropout_prob = 0.5
            model = self.trl_model_class.from_pretrained(pretrained_model)
            assert model.v_head.dropout.p == pretrained_model.config.summary_dropout_prob

    def test_dropout_kwargs(self):
        """
        Test if we instantiate a model by adding `summary_drop_prob` to the config
        it will be added to the v_head
        """
        for model_name in self.all_model_names:
            v_head_kwargs = {'summary_dropout_prob': 0.5}
            model = self.trl_model_class.from_pretrained(model_name, **v_head_kwargs)
            assert model.v_head.dropout.p == 0.5
            model = self.trl_model_class.from_pretrained(model_name, summary_dropout_prob=0.5)
            assert model.v_head.dropout.p == 0.5

    def test_generate(self):
        """
        Test if `generate` works for every model
        """
        for model_name in self.all_model_names:
            model = self.trl_model_class.from_pretrained(model_name)
            input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            decoder_input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            _ = model.generate(input_ids, decoder_input_ids=decoder_input_ids)

    def test_raise_error_not_causallm(self):
        model_id = 'trl-internal-testing/tiny-random-T5Model'
        with pytest.raises(ValueError):
            pretrained_model = AutoModel.from_pretrained(model_id)
            _ = self.trl_model_class.from_pretrained(pretrained_model)

    @unittest.skip('This test needs to be run manually due to HF token issue.')
    def test_push_to_hub(self):
        for model_name in self.all_model_names:
            model = self.trl_model_class.from_pretrained(model_name)
            if 'sharded' in model_name:
                model.push_to_hub(model_name + '-ppo', use_auth_token=True, max_shard_size='1MB')
            else:
                model.push_to_hub(model_name + '-ppo', use_auth_token=True)
            model_from_pretrained = self.trl_model_class.from_pretrained(model_name + '-ppo')
            assert model.state_dict().keys() == model_from_pretrained.state_dict().keys()
            for name, param in model.state_dict().items():
                assert torch.allclose(param, model_from_pretrained.state_dict()[name]), f'Parameter {name} is not the same after push_to_hub and from_pretrained'

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