import copy
import datasets
def tokenize_add_label(sample):
    prompt = tokenizer.encode(tokenizer.bos_token + sample['prompt'], add_special_tokens=False)
    summary = tokenizer.encode(sample['summary'] + tokenizer.eos_token, add_special_tokens=False)
    sample = {'input_ids': prompt + summary, 'attention_mask': [1] * (len(prompt) + len(summary)), 'labels': [-100] * len(prompt) + summary}
    return sample