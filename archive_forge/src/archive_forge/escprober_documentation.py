from typing import Optional, Union
from .charsetprober import CharSetProber
from .codingstatemachine import CodingStateMachine
from .enums import LanguageFilter, MachineState, ProbingState
from .escsm import (

    This CharSetProber uses a "code scheme" approach for detecting encodings,
    whereby easily recognizable escape or shift sequences are relied on to
    identify these encodings.
    