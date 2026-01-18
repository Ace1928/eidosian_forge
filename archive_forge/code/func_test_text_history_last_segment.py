import unittest
from unittest.mock import patch
import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, TextEnvironment, TextHistory
def test_text_history_last_segment(self):
    text = 'Hello there!'
    tokens = torch.tensor([1, 2, 3])
    history = TextHistory(text, tokens)
    history.append_segment('General Kenobi!', torch.tensor([4, 5, 6]))
    history.append_segment('You are a bold one!', torch.tensor([7, 8, 9]))
    assert history.last_text_segment == 'You are a bold one!'