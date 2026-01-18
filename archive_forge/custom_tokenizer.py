from transformers import PreTrainedTokenizer  # type: ignore


class DynamicPreTrainedTokenizer(PreTrainedTokenizer):
    def __init__(self, dynamic_tokenizer, **kwargs):
        """
        Wrap a dynamic tokenizer (e.g. an instance of DynamicTokenizer from intelligent_tokenizer.py)
        into a Hugging Face–compatible tokenizer.

        This implementation inherits special tokens from the dynamic tokenizer (if available) and ensures
        that the tokens are unique by filtering duplicates. Defaults to ["[CLS]", "[SEP]", "[PAD]", "[UNK]"]
        when no special tokens are provided.
        """
        self.dynamic_tokenizer = dynamic_tokenizer
        default_special_tokens = ["[CLS]", "[SEP]", "[PAD]", "[UNK]"]
        tokens = getattr(dynamic_tokenizer, "special_tokens", default_special_tokens)
        self.special_tokens = (
            tuple(dict.fromkeys(tokens)) if tokens else tuple(default_special_tokens)
        )
        self.cls_token = (
            self.special_tokens[0] if len(self.special_tokens) > 0 else "[CLS]"
        )
        self.sep_token = (
            self.special_tokens[1] if len(self.special_tokens) > 1 else "[SEP]"
        )
        self.pad_token = (
            self.special_tokens[2] if len(self.special_tokens) > 2 else "[PAD]"
        )
        self.unk_token = (
            self.special_tokens[3] if len(self.special_tokens) > 3 else "[UNK]"
        )
        super().__init__(**kwargs)

    def _tokenize(self, text):
        """
        Convert text into a list of tokens.

        Here we first encode the text using the dynamic tokenizer (which returns a list of IDs)
        and then convert each ID back into its corresponding token string using the inverse_vocab.
        """
        token_ids = self.dynamic_tokenizer.encode(text)
        tokens = [
            self.dynamic_tokenizer.inverse_vocab.get(idx, self.unk_token)
            for idx in token_ids
        ]
        return tokens

    def _convert_token_to_id(self, token):
        return self.dynamic_tokenizer.vocab.get(
            token, self.dynamic_tokenizer.vocab.get(self.unk_token, 0)
        )

    def _convert_id_to_token(self, index):
        return self.dynamic_tokenizer.inverse_vocab.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """
        Reconstruct a string from a list of tokens. This simple implementation
        simply joins tokens together.
        """
        return "".join(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Adds special tokens (e.g. CLS and SEP) in a robust manner without duplicates.

        Args:
            token_ids_0: The primary sequence of token IDs.
            token_ids_1: Optional secondary sequence of token IDs.

        Returns:
            The list of token IDs with special tokens added.
        """
        cls_token_id = self.dynamic_tokenizer.vocab.get(self.cls_token, 101)
        sep_token_id = self.dynamic_tokenizer.vocab.get(self.sep_token, 102)
        if token_ids_1 is None:
            return [cls_token_id] + token_ids_0 + [sep_token_id]
        else:
            return (
                [cls_token_id]
                + token_ids_0
                + [sep_token_id]
                + token_ids_1
                + [sep_token_id]
            )

    def get_vocab(self):
        return self.dynamic_tokenizer.vocab.copy()
