import unittest
from unittest.mock import patch
import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, TextEnvironment, TextHistory
def test_text_environment_generate(self):
    generation_kwargs = {'do_sample': False, 'max_new_tokens': 4, 'pad_token_id': self.gpt2_tokenizer.eos_token_id}
    env = TextEnvironment(self.gpt2_model, self.gpt2_tokenizer, tools=[DummyTool()], reward_fn=lambda x: torch.tensor(1), prompt='I am a prompt!\n', generation_kwargs=generation_kwargs)
    input_texts = ['this is a test', 'this is another, longer test']
    model_inputs = [self.gpt2_tokenizer(txt, return_tensors='pt').input_ids.squeeze() for txt in input_texts]
    generations_batched = env._generate_batched(model_inputs, batch_size=2)
    generations_batched = self.gpt2_tokenizer.batch_decode(generations_batched)
    generations_single = [env._generate_batched([inputs], batch_size=1)[0] for inputs in model_inputs]
    generations_single = self.gpt2_tokenizer.batch_decode(generations_single)
    assert generations_single == generations_batched