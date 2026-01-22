from sys import version_info as _swig_python_version_info
import re
import csv
import sys
import os
from io import StringIO
from io import BytesIO
from ._version import __version__
def SampleEncodeAndScore(self, input, out_type=None, add_bos=None, add_eos=None, reverse=None, emit_unk_piece=None, num_samples=None, alpha=None, wor=None, include_best=None):
    """SampleEncodeAndScore text input to segmented ids or tokens.

        Args:
        input: input string. accepsts list of string.
        out_type: output type. int or str or 'serialized_proto' or 'immutable_proto'
        add_bos: Add <s> to the result (Default = false)
        add_eos: Add </s> to the result (Default = false) <s>/</s> is added after reversing (if enabled).
        reverse: Reverses the tokenized sequence (Default = false)
        emit_unk_piece: Emits the unk literal string (Default = false)
        num_samples: How many samples to return (Default = 1)
        alpha: inverse temperature for sampling
        wor: whether to sample without replacement (Default = false)
        include_best: whether to include the best tokenization, requires wor=True (Default = false)
      """
    if out_type is None:
        out_type = self._out_type
    if add_bos is None:
        add_bos = self._add_bos
    if add_eos is None:
        add_eos = self._add_eos
    if reverse is None:
        reverse = self._reverse
    if emit_unk_piece is None:
        emit_unk_piece = self._emit_unk_piece
    if num_samples is None:
        num_samples = 1
    if alpha is None:
        alpha = 1.0
    if wor is None:
        wor = False
    if include_best is None:
        include_best = False
    if num_samples <= 0:
        raise RuntimeError('num_examples must be positive')
    if include_best and (not wor):
        raise RuntimeError('When include_best is True, We must specify "wor = True".')

    def _encode(text):
        if out_type is int:
            return self._SampleEncodeAndScoreAsIds(text, num_samples, alpha, wor, include_best, add_bos, add_eos, reverse, emit_unk_piece)
        if out_type is str:
            return self._SampleEncodeAndScoreAsPieces(text, num_samples, alpha, wor, include_best, add_bos, add_eos, reverse, emit_unk_piece)
        if out_type == 'serialized_proto' or out_type == 'proto':
            return self._SampleEncodeAndScoreAsSerializedProto(text, num_samples, alpha, wor, include_best, add_bos, add_eos, reverse, emit_unk_piece)
        if out_type == 'immutable_proto':
            return self._SampleEncodeAndScoreAsImmutableProto(text, num_samples, alpha, wor, include_best, add_bos, add_eos, reverse, emit_unk_piece)
        raise RuntimeError('unknown output type')
    if type(input) is list:
        return [_encode(n) for n in input]
    return _encode(input)