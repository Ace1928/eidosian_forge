import copy
import json
import torch
from torch.utils.data import Dataset
class InstructionDataset(Dataset):

    def __init__(self, dataset_config, tokenizer, partition='train'):
        self.ann = json.load(open(dataset_config.data_path))
        if partition == 'train':
            self.ann = self.ann[200:]
        else:
            self.ann = self.ann[:200]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100
        ann = self.ann[index]
        if ann.get('input', '') == '':
            prompt = PROMPT_DICT['prompt_no_input'].format_map(ann)
        else:
            prompt = PROMPT_DICT['prompt_input'].format_map(ann)
        example = prompt + ann['output']
        prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        labels = copy.deepcopy(example)
        labels[:len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        return {'input_ids': example.tolist(), 'labels': labels.tolist(), 'attention_mask': example_mask.tolist()}