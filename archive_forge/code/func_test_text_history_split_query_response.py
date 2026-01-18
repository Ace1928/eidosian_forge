import unittest
from unittest.mock import patch
import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, TextEnvironment, TextHistory
def test_text_history_split_query_response(self):
    text = 'Hello there!'
    tokens = torch.tensor([1, 2, 3])
    history = TextHistory(text, tokens)
    history.append_segment('General Kenobi!', torch.tensor([4, 5, 6]), system=False)
    history.append_segment('You are a bold one!', torch.tensor([7, 8, 9]), system=True)
    query, response, mask = history.split_query_response_tokens()
    assert torch.equal(query, torch.tensor([1, 2, 3]))
    assert torch.equal(response, torch.tensor([4, 5, 6, 7, 8, 9]))
    assert torch.equal(mask, torch.tensor([1, 1, 1, 0, 0, 0]))