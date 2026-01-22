from parlai.core.dict import DictionaryAgent
from abc import ABC, abstractmethod
class HuggingFaceDictionaryAgent(DictionaryAgent, ABC):
    """
    Use Hugging Face tokenizers.
    """

    def __init__(self, opt):
        super().__init__(opt)
        self.tokenizer = self.get_tokenizer(opt)
        self.override_special_tokens(opt)

    @abstractmethod
    def get_tokenizer(self, opt):
        """
        Instantiate the HuggingFace tokenizer for your model.
        """
        pass

    @abstractmethod
    def override_special_tokens(opt):
        """
        Override the special tokens for your tokenizer.
        """
        pass

    def txt2vec(self, text, vec_type=list):
        tokens = self.tokenizer.tokenize(text)
        tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
        return tokens_id

    def vec2txt(self, vec):
        return self.tokenizer.decode(vec, clean_up_tokenization_spaces=True)

    def act(self):
        """
        Dummy override.
        """
        return {}