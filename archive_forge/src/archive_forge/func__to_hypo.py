from __future__ import annotations
import itertools as it
from abc import abstractmethod
from collections import namedtuple
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
import torch
from flashlight.lib.text.decoder import (
from flashlight.lib.text.dictionary import (
from torchaudio.utils import download_asset
def _to_hypo(self, results) -> List[CTCHypothesis]:
    return [CTCHypothesis(tokens=self._get_tokens(result.tokens), words=[self.word_dict.get_entry(x) for x in result.words if x >= 0], score=result.score, timesteps=self._get_timesteps(result.tokens)) for result in results]