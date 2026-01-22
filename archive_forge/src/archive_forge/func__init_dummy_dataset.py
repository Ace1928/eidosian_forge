import tempfile
import unittest
import torch
from datasets import Dataset
from parameterized import parameterized
from pytest import mark
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from .testing_utils import require_bitsandbytes, require_no_wandb, require_peft
def _init_dummy_dataset(self):
    dummy_dataset_dict = {'prompt': ['hello', 'how are you', 'What is your name?', 'What is your name?', 'Which is the best programming language?', 'Which is the best programming language?', 'Which is the best programming language?', '[INST] How is the stock price? [/INST]', '[INST] How is the stock price? [/INST] '], 'chosen': ['hi nice to meet you', 'I am fine', 'My name is Mary', 'My name is Mary', 'Python', 'Python', 'Python', '$46 as of 10am EST', '46 as of 10am EST'], 'rejected': ['leave me alone', 'I am not fine', 'Whats it to you?', 'I dont have a name', 'Javascript', 'C++', 'Java', ' $46 as of 10am EST', ' 46 as of 10am EST']}
    return Dataset.from_dict(dummy_dataset_dict)