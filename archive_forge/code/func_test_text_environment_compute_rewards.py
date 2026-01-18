import unittest
from unittest.mock import patch
import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, TextEnvironment, TextHistory
def test_text_environment_compute_rewards(self):
    env = TextEnvironment(self.gpt2_model, self.gpt2_tokenizer, tools={'DummyTool': DummyTool()}, reward_fn=lambda x: [torch.tensor(i) for i, _ in enumerate(x)], prompt='I am a prompt!\n')
    histories = [TextHistory('<request><DummyTool>test<call>', torch.tensor([1, 2, 3])) for _ in range(8)]
    histories = env.compute_reward(histories)
    for i in range(8):
        assert histories[i].reward == i