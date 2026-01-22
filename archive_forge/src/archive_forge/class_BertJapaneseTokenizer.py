import collections
import copy
import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import is_sentencepiece_available, is_sudachi_projection_available, logging
class BertJapaneseTokenizer(PreTrainedTokenizer):
    """
    Construct a BERT tokenizer for Japanese text.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer
    to: this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to a one-wordpiece-per-line vocabulary file.
        spm_file (`str`, *optional*):
            Path to [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .spm or .model
            extension) that contains the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether to lower case the input. Only has an effect when do_basic_tokenize=True.
        do_word_tokenize (`bool`, *optional*, defaults to `True`):
            Whether to do word tokenization.
        do_subword_tokenize (`bool`, *optional*, defaults to `True`):
            Whether to do subword tokenization.
        word_tokenizer_type (`str`, *optional*, defaults to `"basic"`):
            Type of word tokenizer. Choose from ["basic", "mecab", "sudachi", "jumanpp"].
        subword_tokenizer_type (`str`, *optional*, defaults to `"wordpiece"`):
            Type of subword tokenizer. Choose from ["wordpiece", "character", "sentencepiece",].
        mecab_kwargs (`dict`, *optional*):
            Dictionary passed to the `MecabTokenizer` constructor.
        sudachi_kwargs (`dict`, *optional*):
            Dictionary passed to the `SudachiTokenizer` constructor.
        jumanpp_kwargs (`dict`, *optional*):
            Dictionary passed to the `JumanppTokenizer` constructor.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file, spm_file=None, do_lower_case=False, do_word_tokenize=True, do_subword_tokenize=True, word_tokenizer_type='basic', subword_tokenizer_type='wordpiece', never_split=None, unk_token='[UNK]', sep_token='[SEP]', pad_token='[PAD]', cls_token='[CLS]', mask_token='[MASK]', mecab_kwargs=None, sudachi_kwargs=None, jumanpp_kwargs=None, **kwargs):
        if subword_tokenizer_type == 'sentencepiece':
            if not os.path.isfile(spm_file):
                raise ValueError(f"Can't find a vocabulary file at path '{spm_file}'. To load the vocabulary from a Google pretrained model use `tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`")
            self.spm_file = spm_file
        else:
            if not os.path.isfile(vocab_file):
                raise ValueError(f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained model use `tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`")
            self.vocab = load_vocab(vocab_file)
            self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_word_tokenize = do_word_tokenize
        self.word_tokenizer_type = word_tokenizer_type
        self.lower_case = do_lower_case
        self.never_split = never_split
        self.mecab_kwargs = copy.deepcopy(mecab_kwargs)
        self.sudachi_kwargs = copy.deepcopy(sudachi_kwargs)
        self.jumanpp_kwargs = copy.deepcopy(jumanpp_kwargs)
        if do_word_tokenize:
            if word_tokenizer_type == 'basic':
                self.word_tokenizer = BasicTokenizer(do_lower_case=do_lower_case, never_split=never_split, tokenize_chinese_chars=False)
            elif word_tokenizer_type == 'mecab':
                self.word_tokenizer = MecabTokenizer(do_lower_case=do_lower_case, never_split=never_split, **mecab_kwargs or {})
            elif word_tokenizer_type == 'sudachi':
                self.word_tokenizer = SudachiTokenizer(do_lower_case=do_lower_case, never_split=never_split, **sudachi_kwargs or {})
            elif word_tokenizer_type == 'jumanpp':
                self.word_tokenizer = JumanppTokenizer(do_lower_case=do_lower_case, never_split=never_split, **jumanpp_kwargs or {})
            else:
                raise ValueError(f"Invalid word_tokenizer_type '{word_tokenizer_type}' is specified.")
        self.do_subword_tokenize = do_subword_tokenize
        self.subword_tokenizer_type = subword_tokenizer_type
        if do_subword_tokenize:
            if subword_tokenizer_type == 'wordpiece':
                self.subword_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))
            elif subword_tokenizer_type == 'character':
                self.subword_tokenizer = CharacterTokenizer(vocab=self.vocab, unk_token=str(unk_token))
            elif subword_tokenizer_type == 'sentencepiece':
                self.subword_tokenizer = SentencepieceTokenizer(vocab=self.spm_file, unk_token=str(unk_token))
            else:
                raise ValueError(f"Invalid subword_tokenizer_type '{subword_tokenizer_type}' is specified.")
        super().__init__(spm_file=spm_file, unk_token=unk_token, sep_token=sep_token, pad_token=pad_token, cls_token=cls_token, mask_token=mask_token, do_lower_case=do_lower_case, do_word_tokenize=do_word_tokenize, do_subword_tokenize=do_subword_tokenize, word_tokenizer_type=word_tokenizer_type, subword_tokenizer_type=subword_tokenizer_type, never_split=never_split, mecab_kwargs=mecab_kwargs, sudachi_kwargs=sudachi_kwargs, jumanpp_kwargs=jumanpp_kwargs, **kwargs)

    @property
    def do_lower_case(self):
        return self.lower_case

    def __getstate__(self):
        state = dict(self.__dict__)
        if self.word_tokenizer_type in ['mecab', 'sudachi', 'jumanpp']:
            del state['word_tokenizer']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        if self.word_tokenizer_type == 'mecab':
            self.word_tokenizer = MecabTokenizer(do_lower_case=self.do_lower_case, never_split=self.never_split, **self.mecab_kwargs or {})
        elif self.word_tokenizer_type == 'sudachi':
            self.word_tokenizer = SudachiTokenizer(do_lower_case=self.do_lower_case, never_split=self.never_split, **self.sudachi_kwargs or {})
        elif self.word_tokenizer_type == 'jumanpp':
            self.word_tokenizer = JumanppTokenizer(do_lower_case=self.do_lower_case, never_split=self.never_split, **self.jumanpp_kwargs or {})

    def _tokenize(self, text):
        if self.do_word_tokenize:
            tokens = self.word_tokenizer.tokenize(text, never_split=self.all_special_tokens)
        else:
            tokens = [text]
        if self.do_subword_tokenize:
            split_tokens = [sub_token for token in tokens for sub_token in self.subword_tokenizer.tokenize(token)]
        else:
            split_tokens = tokens
        return split_tokens

    @property
    def vocab_size(self):
        if self.subword_tokenizer_type == 'sentencepiece':
            return len(self.subword_tokenizer.sp_model)
        return len(self.vocab)

    def get_vocab(self):
        if self.subword_tokenizer_type == 'sentencepiece':
            vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
            vocab.update(self.added_tokens_encoder)
            return vocab
        return dict(self.vocab, **self.added_tokens_encoder)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if self.subword_tokenizer_type == 'sentencepiece':
            return self.subword_tokenizer.sp_model.PieceToId(token)
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if self.subword_tokenizer_type == 'sentencepiece':
            return self.subword_tokenizer.sp_model.IdToPiece(index)
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        if self.subword_tokenizer_type == 'sentencepiece':
            return self.subword_tokenizer.sp_model.decode(tokens)
        out_string = ' '.join(tokens).replace(' ##', '').strip()
        return out_string

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None, already_has_special_tokens: bool=False) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True)
        if token_ids_1 is not None:
            return [1] + [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]
        return [1] + [0] * len(token_ids_0) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str]=None) -> Tuple[str]:
        if os.path.isdir(save_directory):
            if self.subword_tokenizer_type == 'sentencepiece':
                vocab_file = os.path.join(save_directory, (filename_prefix + '-' if filename_prefix else '') + VOCAB_FILES_NAMES['spm_file'])
            else:
                vocab_file = os.path.join(save_directory, (filename_prefix + '-' if filename_prefix else '') + VOCAB_FILES_NAMES['vocab_file'])
        else:
            vocab_file = (filename_prefix + '-' if filename_prefix else '') + save_directory
        if self.subword_tokenizer_type == 'sentencepiece':
            with open(vocab_file, 'wb') as writer:
                content_spiece_model = self.subword_tokenizer.sp_model.serialized_model_proto()
                writer.write(content_spiece_model)
        else:
            with open(vocab_file, 'w', encoding='utf-8') as writer:
                index = 0
                for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                    if index != token_index:
                        logger.warning(f'Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive. Please check that the vocabulary is not corrupted!')
                        index = token_index
                    writer.write(token + '\n')
                    index += 1
        return (vocab_file,)