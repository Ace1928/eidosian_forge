from parlai.core.dict import DictionaryAgent
from abc import ABC, abstractmethod
class Gpt2DictionaryAgent(HuggingFaceDictionaryAgent):

    def is_prebuilt(self):
        """
        Indicates whether the dictionary is fixed, and does not require building.
        """
        return True

    def get_tokenizer(self, opt):
        """
        Instantiate tokenizer.
        """
        model_sz = opt['gpt2_size']
        fle_key = 'gpt2' if model_sz == 'small' else f'gpt2-{model_sz}'
        return GPT2Tokenizer.from_pretrained(fle_key)

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

    def override_special_tokens(self, opt):
        self._define_special_tokens(opt)
        self.start_idx = self.tokenizer.convert_tokens_to_ids([self.start_token])[0]
        self.end_idx = self.tokenizer.convert_tokens_to_ids([self.end_token])[0]
        self.null_idx = self.tokenizer.convert_tokens_to_ids([self.null_token])[0]
        self.tok2ind[self.end_token] = self.end_idx
        self.tok2ind[self.start_token] = self.start_idx
        self.tok2ind[self.null_token] = self.null_idx
        self.ind2tok[self.end_idx] = self.end_token
        self.ind2tok[self.start_idx] = self.start_token
        self.ind2tok[self.null_idx] = self.null_token