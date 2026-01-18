import argparse
import os
import warnings
import torch
from accelerate import init_empty_weights
from transformers import GemmaConfig, GemmaForCausalLM, GemmaTokenizer
from transformers import GemmaForCausalLM, GemmaTokenizerFast
def write_tokenizer(input_tokenizer_path, save_path, push_to_hub=False):
    tokenizer_class = GemmaTokenizer if GemmaTokenizerFast is None else GemmaTokenizerFast
    print(f'Saving a {tokenizer_class.__name__} to {save_path}.')
    tokenizer = tokenizer_class(input_tokenizer_path)
    if push_to_hub:
        tokenizer.push_to_hub(save_path)
    else:
        tokenizer.save_pretrained(save_path)