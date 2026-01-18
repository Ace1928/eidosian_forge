import json
import os
from typing import List, Optional, Tuple, Union
import numpy as np
from ...feature_extraction_utils import BatchFeature
from ...tokenization_utils import AddedToken, BatchEncoding, PaddingStrategy, PreTrainedTokenizer, TruncationStrategy
from ...utils import TensorType, is_pretty_midi_available, logging, requires_backends, to_numpy
def token_note_to_note(number, current_velocity, default_velocity, note_onsets_ready, current_idx, notes):
    if note_onsets_ready[number] is not None:
        onset_idx = note_onsets_ready[number]
        if onset_idx < current_idx:
            offset_idx = current_idx
            notes.append([onset_idx, offset_idx, number, default_velocity])
            onsets_ready = None if current_velocity == 0 else current_idx
            note_onsets_ready[number] = onsets_ready
    else:
        note_onsets_ready[number] = current_idx
    return notes