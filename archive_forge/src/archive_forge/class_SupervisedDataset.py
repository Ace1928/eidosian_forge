import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning('Loading data...')
        list_data_dict = utils.jload(data_path)
        logging.warning('Formatting inputs...')
        prompt_input, prompt_no_input = (PROMPT_DICT['prompt_input'], PROMPT_DICT['prompt_no_input'])
        sources = [prompt_input.format_map(example) if example.get('input', '') != '' else prompt_no_input.format_map(example) for example in list_data_dict]
        targets = [f'{example['output']}{tokenizer.eos_token}' for example in list_data_dict]
        logging.warning('Tokenizing inputs... This may take some time...')
        data_dict = preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict['input_ids']
        self.labels = data_dict['labels']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])