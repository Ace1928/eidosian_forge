import collections
import json
import os
import re
from typing import List, Optional, Tuple, Union
import numpy as np
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import (
from ...utils import PaddingStrategy, logging
class GPTSanJapaneseTokenizer(PreTrainedTokenizer):
    """
    This tokenizer is based on GPTNeoXJapaneseTokenizer and has the following modifications
    - Decoding byte0~byte255 tokens correctly
    - Added bagofword token handling
    - Return token_type_ids for Prefix-LM model
    The bagofword token represents a repetition of the previous token and is converted to 3 consecutive tokens when
    decoding In addition, the original Japanese special Sub-Word-Encoding has been released in this repository
    (https://github.com/tanreinama/Japanese-BPEEncoder_V2). The token_type_ids is a mask indicating the prefix input
    position of the Prefix-LM model. To specify a prefix position, specify a prefix input for prefix_text, or specify a
    sentence of the prefix part and the part after it as a text pair of batch input.

    Example:

    ```python
    >>> from transformers import GPTSanJapaneseTokenizer

    >>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
    >>> # You can confirm both 慶応 and 慶應 are encoded to 17750
    >>> tokenizer("吾輩は猫である🐯。実は慶応(慶應)大学出身")["input_ids"]
    [35993, 35998, 34347, 31459, 30647, 31448, 25, 30659, 35729, 35676, 32417, 30647, 17750, 35589, 17750, 35590, 321, 1281]

    >>> # Both 慶応 and 慶應 are decoded to 慶応
    >>> tokenizer.decode(tokenizer("吾輩は猫である🐯。実は慶応(慶應)大学出身")["input_ids"])
    '吾輩は猫である🐯。実は慶応(慶応)大学出身'
    ```

    Example for Prefix-LM:

    ```python
    >>> from transformers import GPTSanJapaneseTokenizer

    >>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
    >>> tokenizer("実は慶応(慶應)大学出身", prefix_text="吾輩は猫である🐯。")["input_ids"]
    [35993, 34347, 31459, 30647, 31448, 25, 30659, 35729, 35676, 35998, 32417, 30647, 17750, 35589, 17750, 35590, 321, 1281]

    >>> # Mask for Prefix-LM inputs
    >>> tokenizer("実は慶応(慶應)大学出身", prefix_text="吾輩は猫である🐯。")["token_type_ids"]
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ```

    Example for batch encode:

    ```python
    >>> from transformers import GPTSanJapaneseTokenizer

    >>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
    >>> tokenizer([["武田信玄", "は、"], ["織田信長", "の配下の、"]], padding=True)["input_ids"]
    [[35993, 8640, 25948, 35998, 30647, 35675, 35999, 35999], [35993, 10382, 9868, 35998, 30646, 9459, 30646, 35675]]

    >>> # Mask for Prefix-LM inputs
    >>> tokenizer([["武田信玄", "は、"], ["織田信長", "の配下の、"]], padding=True)["token_type_ids"]
    [[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0]]

    >>> # Mask for padding
    >>> tokenizer([["武田信玄", "は、"], ["織田信長", "の配下の、"]], padding=True)["attention_mask"]
    [[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]]
    ```

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        emoji_file (`str`):
            File containing the emoji.
        unk_token (`str`, *optional*, defaults to `"<|nottoken|>"`):
            The token used for unknown charactor
        pad_token (`str`, *optional*, defaults to `"<|separator|>"`):
            The token used for padding
        bos_token (`str`, *optional*, defaults to `"<|startoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        sep_token (`str`, *optional*, defaults to `"<|segmenter|>"`):
            A special token to separate token to prefix part and general input part.
        do_clean_text (`bool`, *optional*, defaults to `False`):
            Whether or not to clean text for URL, EMAIL, TEL, Japanese DATE and Japanese PRICE.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ['input_ids', 'attention_mask', 'token_type_ids']

    def __init__(self, vocab_file, emoji_file, unk_token='<|nottoken|>', pad_token='<|separator|>', bos_token='<|startoftext|>', eos_token='<|endoftext|>', sep_token='<|segmenter|>', do_clean_text=False, **kwargs):
        if not os.path.isfile(vocab_file):
            raise ValueError(f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained model use `tokenizer = GPTSanJapaneseTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`")
        if not os.path.isfile(emoji_file):
            raise ValueError(f"Can't find a emoji file at path '{emoji_file}'. To load the emoji information from a Google pretrained model use `tokenizer = GPTSanJapaneseTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`")
        self.do_clean_text = do_clean_text
        self.vocab, self.raw_vocab, self.ids_to_tokens, self.emoji = load_vocab_and_emoji(vocab_file, emoji_file)
        self.subword_tokenizer = SubWordJapaneseTokenizer(vocab=self.vocab, ids_to_tokens=self.ids_to_tokens, emoji=self.emoji)
        super().__init__(unk_token=unk_token, pad_token=pad_token, bos_token=bos_token, eos_token=eos_token, sep_token=sep_token, do_clean_text=do_clean_text, **kwargs)

    @property
    def vocab_size(self):
        return len(self.raw_vocab)

    def get_vocab(self):
        return dict(self.raw_vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        return self.subword_tokenizer.tokenize(text, clean=self.do_clean_text)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.subword_tokenizer.convert_id_to_token(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        words = []
        byte_tokens = []
        for word in tokens:
            if word[:6] == '<|byte' and word[-2:] == '|>':
                byte_tokens.append(int(word[6:-2]))
            else:
                if len(byte_tokens) > 0:
                    words.append(bytearray(byte_tokens).decode('utf-8', errors='replace'))
                    byte_tokens = []
                if word[:7] == '<|emoji' and word[-2:] == '|>':
                    words.append(self.emoji['emoji_inv'][word])
                elif word == '<SP>':
                    words.append(' ')
                elif word == '<BR>':
                    words.append('\n')
                elif word == '<TAB>':
                    words.append('\t')
                elif word == '<BLOCK>':
                    words.append('▀')
                elif word == '<KIGOU>':
                    words.append('ǀ')
                elif word == '<U2000U2BFF>':
                    words.append('‖')
                elif word == '<|bagoftoken|>':
                    if len(words) > 0:
                        words.append(words[-1])
                        words.append(words[-1])
                        words.append(words[-1])
                elif word.startswith('<|') and word.endswith('|>'):
                    words.append('')
                else:
                    words.append(word)
        if len(byte_tokens) > 0:
            words.append(bytearray(byte_tokens).decode('utf-8', errors='replace'))
        text = ''.join(words)
        return text

    @property
    def default_chat_template(self):
        """
        A simple chat template that adds standard BOS, SEP and EOS tokens between messages while discarding role
        information.
        """
        logger.warning_once(f'\nNo chat template is defined for this tokenizer - using the default template for the {self.__class__.__name__} class. If the default is not appropriate for your model, please set `tokenizer.chat_template` to an appropriate template. See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n')
        return '{% for message in messages %}{% if not loop.first %}{{ bos_token}}{% endif %}{{ sep_token }}{{ message.content }} {{ eos_token }}{% endfor %}'

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str]=None) -> Tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(save_directory, (filename_prefix + '-' if filename_prefix else '') + VOCAB_FILES_NAMES['vocab_file'])
            emoji_file = os.path.join(save_directory, (filename_prefix + '-' if filename_prefix else '') + VOCAB_FILES_NAMES['emoji_file'])
        else:
            vocab_file = (filename_prefix + '-' if filename_prefix else '') + save_directory + VOCAB_FILES_NAMES['vocab_file']
            emoji_file = (filename_prefix + '-' if filename_prefix else '') + save_directory + VOCAB_FILES_NAMES['emoji_file']
        with open(vocab_file, 'w', encoding='utf-8') as writer:
            for token_index, token in self.ids_to_tokens.items():
                if index != token_index:
                    logger.warning(f'Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive. Please check that the vocabulary is not corrupted!')
                    index = token_index
                writer.write(','.join(token) + '\n')
                index += 1
        with open(emoji_file, 'w', encoding='utf-8') as writer:
            json.dump(self.emoji, writer)
        return (vocab_file, emoji_file)

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        """
        The tokenizer returns token_type_ids as separators between the Prefix part and the rest.
        token_type_ids is 1 for the Prefix part and 0 for the rest of the token.

        Example:
        ```python
        >>> from transformers import GPTSanJapaneseTokenizer

        >>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
        >>> x_token = tokenizer("ｱｲｳｴ")
        >>> # input_ids:      | SOT | SEG | ｱ | ｲ | ｳ | ｴ |
        >>> # token_type_ids: | 1   | 0   | 0 | 0 | 0 | 0 |

        >>> x_token = tokenizer("", prefix_text="ｱｲｳｴ")
        >>> # input_ids:      | SOT | ｱ | ｲ | ｳ | ｴ | SEG |
        >>> # token_type_ids: | 1   | 1 | 1 | 1 | 1 | 0  |

        >>> x_token = tokenizer("ｳｴ", prefix_text="ｱｲ")
        >>> # input_ids:      | SOT | ｱ | ｲ | SEG | ｳ | ｴ |
        >>> # token_type_ids: | 1   | 1 | 1 | 0   | 0 | 0 |
        ```"""
        prefix_len = 0
        if self.sep_token in self.vocab:
            segid = self.vocab[self.sep_token]
            if segid in token_ids_0:
                prefix_len = token_ids_0.index(segid)
        if token_ids_1 is None:
            total_len = len(token_ids_0)
        else:
            total_len = len(token_ids_0 + token_ids_1)
        return prefix_len * [1] + (total_len - prefix_len) * [0]

    def prepare_for_tokenization(self, text, prefix_text=None, add_sep_token=None, **kwargs):
        if add_sep_token is None:
            add_sep_token = self.sep_token not in text
        prepared = self.bos_token if self.bos_token in self.vocab else ''
        prepared += prefix_text if prefix_text is not None else ''
        if add_sep_token:
            prepared += self.sep_token if self.sep_token in self.vocab else ''
        prepared += text
        return (prepared, kwargs)

    def _batch_encode_plus(self, batch_text_or_text_pairs: Union[List[TextInput], List[TextInputPair], List[PreTokenizedInput], List[PreTokenizedInputPair]], add_special_tokens: bool=True, padding_strategy: PaddingStrategy=PaddingStrategy.DO_NOT_PAD, truncation_strategy: TruncationStrategy=TruncationStrategy.DO_NOT_TRUNCATE, max_length: Optional[int]=None, stride: int=0, is_split_into_words: bool=False, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[str]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True) -> BatchEncoding:
        if isinstance(batch_text_or_text_pairs[0], tuple) or isinstance(tuple(batch_text_or_text_pairs[0]), list):
            batch_prefix_texts = []
            for pref, txt in batch_text_or_text_pairs:
                batch_prefix_texts.append(pref + self.sep_token + txt)
            batch_text_or_text_pairs = batch_prefix_texts
        return super()._batch_encode_plus(batch_text_or_text_pairs, add_special_tokens, padding_strategy, truncation_strategy, max_length, stride, is_split_into_words, pad_to_multiple_of, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose)