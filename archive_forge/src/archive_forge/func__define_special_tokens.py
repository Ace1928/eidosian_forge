from parlai.core.dict import DictionaryAgent
from abc import ABC, abstractmethod
def _define_special_tokens(self, opt):
    if opt['add_special_tokens']:
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)
        self.start_token = SPECIAL_TOKENS['bos_token']
        self.end_token = SPECIAL_TOKENS['eos_token']
        self.null_token = SPECIAL_TOKENS['pad_token']
    else:
        self.start_token = NO_OP
        self.end_token = '<|endoftext|>'
        self.null_token = '<|endoftext|>'