import unittest
from unittest.mock import patch
import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, TextEnvironment, TextHistory
def test_text_environment_setup(self):
    env = TextEnvironment(self.gpt2_model, self.gpt2_tokenizer, tools=[DummyTool()], reward_fn=lambda x: torch.tensor(1), prompt='I am a prompt!\n')
    assert env.prompt == 'I am a prompt!\n'
    assert list(env.tools.keys()) == ['DummyTool']
    assert isinstance(env.tools['DummyTool'], DummyTool)
    assert env.reward_fn('Hello there!') == 1