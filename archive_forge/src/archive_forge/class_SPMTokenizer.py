import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
import sentencepiece as sp
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging
class SPMTokenizer:
    """
    Constructs a tokenizer based on [SentencePiece](https://github.com/google/sentencepiece).

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.
    """

    def __init__(self, vocab_file, special_tokens, split_by_punct=False, sp_model_kwargs: Optional[Dict[str, Any]]=None):
        self.split_by_punct = split_by_punct
        self.vocab_file = vocab_file
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        spm = sp.SentencePieceProcessor(**self.sp_model_kwargs)
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f'{vocab_file} does not exist!')
        spm.load(vocab_file)
        bpe_vocab_size = spm.GetPieceSize()
        self.vocab = {spm.IdToPiece(i): i for i in range(bpe_vocab_size)}
        self.ids_to_tokens = [spm.IdToPiece(i) for i in range(bpe_vocab_size)]
        self.spm = spm
        self.special_tokens = special_tokens

    def __getstate__(self):
        state = self.__dict__.copy()
        state['spm'] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        if not hasattr(self, 'sp_model_kwargs'):
            self.sp_model_kwargs = {}
        self.spm = sp.SentencePieceProcessor(**self.sp_model_kwargs)
        self.spm.Load(self.vocab_file)

    def tokenize(self, text):
        return self._encode_as_pieces(text)

    def convert_ids_to_tokens(self, ids):
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    def decode(self, tokens, start=-1, end=-1, raw_text=None):
        if raw_text is None:
            current_sub_tokens = []
            out_string = ''
            prev_is_special = False
            for token in tokens:
                if token in self.special_tokens:
                    if not prev_is_special:
                        out_string += ' '
                    out_string += self.spm.decode_pieces(current_sub_tokens) + token
                    prev_is_special = True
                    current_sub_tokens = []
                else:
                    current_sub_tokens.append(token)
                    prev_is_special = False
            out_string += self.spm.decode_pieces(current_sub_tokens)
            return out_string.strip()
        else:
            words = self.split_to_words(raw_text)
            word_tokens = [self.tokenize(w) for w in words]
            token2words = [0] * len(tokens)
            tid = 0
            for i, w in enumerate(word_tokens):
                for k, t in enumerate(w):
                    token2words[tid] = i
                    tid += 1
            word_start = token2words[start]
            word_end = token2words[end] if end < len(tokens) else len(words)
            text = ''.join(words[word_start:word_end])
            return text

    def add_special_token(self, token):
        if token not in self.special_tokens:
            self.special_tokens.append(token)
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) - 1
                self.ids_to_tokens.append(token)
        return self.id(token)

    def part_of_whole_word(self, token, is_bos=False):
        logger.warning_once('The `DebertaTokenizer.part_of_whole_word` method is deprecated and will be removed in `transformers==4.35`')
        if is_bos:
            return True
        if len(token) == 1 and (_is_whitespace(list(token)[0]) or _is_control(list(token)[0]) or _is_punctuation(list(token)[0])) or token in self.special_tokens:
            return False
        word_start = b'\xe2\x96\x81'.decode('utf-8')
        return not token.startswith(word_start)

    def pad(self):
        return '[PAD]'

    def bos(self):
        return '[CLS]'

    def eos(self):
        return '[SEP]'

    def unk(self):
        return '[UNK]'

    def mask(self):
        return '[MASK]'

    def sym(self, id):
        return self.ids_to_tokens[id]

    def id(self, sym):
        logger.warning_once('The `DebertaTokenizer.id` method is deprecated and will be removed in `transformers==4.35`')
        return self.vocab[sym] if sym in self.vocab else 1

    def _encode_as_pieces(self, text):
        text = convert_to_unicode(text)
        if self.split_by_punct:
            words = self._run_split_on_punc(text)
            pieces = [self.spm.encode(w, out_type=str) for w in words]
            return [p for w in pieces for p in w]
        else:
            return self.spm.encode(text, out_type=str)

    def split_to_words(self, text):
        pieces = self._encode_as_pieces(text)
        word_start = b'\xe2\x96\x81'.decode('utf-8')
        words = []
        offset = 0
        prev_end = 0
        for i, p in enumerate(pieces):
            if p.startswith(word_start):
                if offset > prev_end:
                    words.append(text[prev_end:offset])
                prev_end = offset
                w = p.replace(word_start, '')
            else:
                w = p
            try:
                s = text.index(w, offset)
                pn = ''
                k = i + 1
                while k < len(pieces):
                    pn = pieces[k].replace(word_start, '')
                    if len(pn) > 0:
                        break
                    k += 1
                if len(pn) > 0 and pn in text[offset:s]:
                    offset = offset + 1
                else:
                    offset = s + len(w)
            except Exception:
                offset = offset + 1
        if prev_end < offset:
            words.append(text[prev_end:offset])
        return words

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return [''.join(x) for x in output]

    def save_pretrained(self, path: str, filename_prefix: str=None):
        filename = VOCAB_FILES_NAMES[list(VOCAB_FILES_NAMES.keys())[0]]
        if filename_prefix is not None:
            filename = filename_prefix + '-' + filename
        full_path = os.path.join(path, filename)
        with open(full_path, 'wb') as fs:
            fs.write(self.spm.serialized_model_proto())
        return (full_path,)