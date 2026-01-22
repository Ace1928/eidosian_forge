from abc import ABC, abstractmethod
from functools import lru_cache
import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple
from typing_extensions import final
from parlai.core.build_data import download, make_dir
from parlai.core.opt import Opt
from parlai.utils.misc import warn_once
from parlai.utils.typing import TShared
import parlai.utils.logging as logging
class SubwordBPEHelper(BPEHelper):
    """
    Helper class for performing BPE subword tokenization.

    For technical details, please refer to https://arxiv.org/abs/1508.07909.
    This class just wraps around the official subword-nmt repository.

    This API expects the user to call tokenize() (encode) onto the training data,
    then call finalize() to learn the encodings, and then iterate over the data
    in a second pass, calling tokenize() again to get processed output.
    """

    def __init__(self, opt: Opt, shared: TShared=None):
        """
        Initialize the BPE module.

        :param opt:
            options
        :param shared:
            shared dictionary
        """
        super().__init__(opt, shared)
        if not SUBWORD_BPE_INSTALLED:
            raise RuntimeError('Please run "pip install \'git+https://github.com/rsennrich/subword-nmt.git#egg=subword-nmt\'"')
        if not opt.get('dict_file'):
            raise RuntimeError('--dict-file is mandatory.')
        self.splitter = re.compile('\\w+|[^\\w\\s]', re.UNICODE)
        self.codecs = f'{opt['dict_file']}.codecs'
        if os.path.exists(self.codecs):
            self._load_from_codecs()

    def helper_encode(self, text: str) -> List[str]:
        """
        Tokenize the text with bpe if codecs are already finalized.

        Otherwise, returns the regularly split tokens that will train the bpe.

        :param text:
            Raw text to tokenize.
        :return:
            a list of tokens. Will use BPE once finalized.
        """
        text = text.replace('\n', ' __newln__ ')
        tokens = self.splitter.findall(text)
        if hasattr(self, 'bpe'):
            return self.bpe.segment_tokens(tokens)
        else:
            return tokens

    def helper_decode(self, tokens: List[str], token_ids: List[int], delimiter: str) -> str:
        """
        Decode list of tokens into text string.

        :param tokens:
            list of tokens
        :param token_ids:
            list of token ids
        :param delimiter:
            string delimiter for tokens

        :return text:
            decoded text
        """
        text = delimiter.join(tokens)
        text = text.replace('@@ ', '')
        if text.endswith('@@'):
            text = text[:-2]
        text = text.replace('__newln__', '\n')
        return text

    def finalize(self, frequencies: Dict[str, int], num_symbols: int=30000, minfreq: int=2) -> bool:
        """
        Build the codecs.

        :param frequencies:
            dictionary of (token: frequency) pairs
        :param num_symbols:
            Number of BPE symbols. Recommend 30000-40000.  If <= 0, default
            30000 will be used.
        :param minfreq:
            Minimum frequency of a token before forced BPE decomposition. If <=
            0 will use subword-nmt default of 2.

        :return did_finalize:
            return whether codecs are finalized this call.
        """
        if hasattr(self, 'bpe'):
            return False
        logging.debug(f'Saving bpe codecs to {self.codecs}')
        dictionary = ('{} {}'.format(k, v) for k, v in frequencies.items())
        if num_symbols <= 0:
            num_symbols = 30000
        if minfreq <= 0:
            minfreq = 2
        codec_dir, _ = os.path.split(self.codecs)
        os.makedirs(codec_dir, exist_ok=True)
        with open(self.codecs, 'w', encoding='utf-8') as outstream:
            learn_bpe.learn_bpe(dictionary, outstream, num_symbols=num_symbols, min_frequency=minfreq, is_dict=True)
        self._load_from_codecs()
        return True

    def _load_from_codecs(self):
        """
        Load BPE from codecs file.
        """
        with open(self.codecs, 'r', encoding='utf-8') as codecs_file:
            self.bpe = apply_bpe.BPE(codecs_file)

    def copy_codecs_file(self, target_file: str):
        """
        Copy the codecs file to a new location.

        :param target_file:
            where to copy the codecs.
        """
        with open(target_file, 'w', encoding='utf-8') as wfile:
            with open(self.codecs, encoding='utf-8') as rfile:
                for line in rfile:
                    wfile.write(line)

    def sync_with_dict(self, dict_agent):
        """
        No need to sync subword BPE.
        """
        pass

    def should_sort(self) -> bool:
        """
        Return whether tokens should be sorted for this particular helper.

        We want to sort with SubwordBPEHelper.
        """
        return True