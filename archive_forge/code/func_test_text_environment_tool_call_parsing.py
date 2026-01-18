import unittest
from unittest.mock import patch
import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, TextEnvironment, TextHistory
def test_text_environment_tool_call_parsing(self):
    string_valid = 'Something something <request><Tool1>Hello there!<call>'
    string_invalid_request = 'Something something <Tool1>Hello there!<call>'
    string_invalid_call = 'Something something <request><Tool1>Hello there!'
    string_invalid_tool = 'Something something <request>|Tool2|Hello there!<call>'
    string_invalid_random = '<>abcdefghijklm<>nopqrstuvwxyz<>'
    env = TextEnvironment(self.gpt2_model, self.gpt2_tokenizer, tools=[DummyTool()], reward_fn=lambda x: torch.tensor(1), prompt='I am a prompt!\n')
    tool, response = env.parse_tool_call(string_valid)
    assert tool == 'Tool1'
    assert response == 'Hello there!'
    tool, response = env.parse_tool_call(string_invalid_request)
    assert tool is None
    assert response is None
    tool, response = env.parse_tool_call(string_invalid_call)
    assert tool is None
    assert response is None
    tool, response = env.parse_tool_call(string_invalid_tool)
    assert tool is None
    assert response is None
    tool, response = env.parse_tool_call(string_invalid_random)
    assert tool is None
    assert response is None