from __future__ import annotations
import itertools as it
from abc import abstractmethod
from collections import namedtuple
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
import torch
from flashlight.lib.text.decoder import (
from flashlight.lib.text.dictionary import (
from torchaudio.utils import download_asset
class CTCDecoderLM(_LM):
    """Language model base class for creating custom language models to use with the decoder."""

    @abstractmethod
    def start(self, start_with_nothing: bool) -> CTCDecoderLMState:
        """Initialize or reset the language model.

        Args:
            start_with_nothing (bool): whether or not to start sentence with sil token.

        Returns:
            CTCDecoderLMState: starting state
        """
        raise NotImplementedError

    @abstractmethod
    def score(self, state: CTCDecoderLMState, usr_token_idx: int) -> Tuple[CTCDecoderLMState, float]:
        """Evaluate the language model based on the current LM state and new word.

        Args:
            state (CTCDecoderLMState): current LM state
            usr_token_idx (int): index of the word

        Returns:
            (CTCDecoderLMState, float)
                CTCDecoderLMState:
                    new LM state
                float:
                    score
        """
        raise NotImplementedError

    @abstractmethod
    def finish(self, state: CTCDecoderLMState) -> Tuple[CTCDecoderLMState, float]:
        """Evaluate end for language model based on current LM state.

        Args:
            state (CTCDecoderLMState): current LM state

        Returns:
            (CTCDecoderLMState, float)
                CTCDecoderLMState:
                    new LM state
                float:
                    score
        """
        raise NotImplementedError