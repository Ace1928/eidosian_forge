import os
import signal
import sys
import pickle
from .exceptions import RestartFreqExceeded
from time import monotonic
from io import BytesIO
def _should_override_term_signal(sig, current):
    return sig in TERMSIGS_FORCE or (current is not None and current != signal.SIG_IGN)