from datasets import load_dataset
from pathlib import Path
from torch.utils.data import Dataset
class grammar(Dataset):

    def __init__(self, tokenizer, csv_name=None):
        try:
            self.dataset = load_dataset('csv', data_files={'train': [csv_name]}, delimiter=',')
        except Exception as e:
            print('Loading of grammar dataset failed! Please see recipes/ft_datasets/grammar_dataset/grammar_dataset_process.ipynb for details on how to download the dataset.')
            raise e
        self.tokenizer = tokenizer
        self.print_text = False

    def __len__(self):
        return self.dataset['train'].shape[0]

    def convert_to_features(self, example_batch):
        if self.print_text:
            print('Input Text: ', self.clean_text(example_batch['text']))
        input_ = example_batch['input']
        target_ = example_batch['target']
        prompt = f'Correct this to standard English: {input_}\n---\nCorrected: '
        prompt_ids = self.tokenizer.encode(self.tokenizer.bos_token + prompt, add_special_tokens=False)
        label_ids = self.tokenizer.encode(target_ + self.tokenizer.eos_token, add_special_tokens=False)
        sample = {'input_ids': prompt_ids + label_ids, 'attention_mask': [1] * len(prompt_ids + label_ids), 'labels': [-100] * len(prompt_ids) + label_ids}
        return sample

    def __getitem__(self, index):
        return self.convert_to_features(self.dataset['train'][int(index)])