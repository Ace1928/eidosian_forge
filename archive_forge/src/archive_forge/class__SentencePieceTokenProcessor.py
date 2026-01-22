import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Tuple
import torch
import torchaudio
from torchaudio._internal import module_utils
from torchaudio.models import emformer_rnnt_base, RNNT, RNNTBeamSearch
class _SentencePieceTokenProcessor(_TokenProcessor):
    """SentencePiece-model-based token processor.

    Args:
        sp_model_path (str): path to SentencePiece model.
    """

    def __init__(self, sp_model_path: str) -> None:
        if not module_utils.is_module_available('sentencepiece'):
            raise RuntimeError('SentencePiece is not available. Please install it.')
        import sentencepiece as spm
        self.sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.post_process_remove_list = {self.sp_model.unk_id(), self.sp_model.eos_id(), self.sp_model.pad_id()}

    def __call__(self, tokens: List[int], lstrip: bool=True) -> str:
        """Decodes given list of tokens to text sequence.

        Args:
            tokens (List[int]): list of tokens to decode.
            lstrip (bool, optional): if ``True``, returns text sequence with leading whitespace
                removed. (Default: ``True``).

        Returns:
            str:
                Decoded text sequence.
        """
        filtered_hypo_tokens = [token_index for token_index in tokens[1:] if token_index not in self.post_process_remove_list]
        output_string = ''.join(self.sp_model.id_to_piece(filtered_hypo_tokens)).replace('▁', ' ')
        if lstrip:
            return output_string.lstrip()
        else:
            return output_string