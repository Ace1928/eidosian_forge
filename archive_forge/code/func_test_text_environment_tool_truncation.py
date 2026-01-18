import unittest
from unittest.mock import patch
import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, TextEnvironment, TextHistory
def test_text_environment_tool_truncation(self):
    env = TextEnvironment(self.gpt2_model, self.gpt2_tokenizer, tools={'dummy': lambda x: 'a' * 1000}, reward_fn=lambda x: torch.tensor(1), prompt='I am a prompt!\n')
    env.max_tool_response = 100
    history = env.step(TextHistory('<request><dummy>Hello there!<call>', torch.tensor([1, 2, 3])))
    assert len(history.last_text_segment) - len(env.response_token) == 100
    env.max_tool_response = 500
    history = env.step(TextHistory('<request><dummy>Hello there!<call>', torch.tensor([1, 2, 3])))
    assert len(history.last_text_segment) - len(env.response_token) == 500
    env.max_tool_response = 1001
    history = env.step(TextHistory('<request><dummy>Hello there!<call>', torch.tensor([1, 2, 3])))
    assert len(history.last_text_segment) - len(env.response_token) == 1000
    env.max_tool_response = 2000
    history = env.step(TextHistory('<request><dummy>Hello there!<call>', torch.tensor([1, 2, 3])))
    assert len(history.last_text_segment) - len(env.response_token) == 1000