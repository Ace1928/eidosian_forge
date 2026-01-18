from pathlib import Path
import numpy as np
import scipy.io
import scipy.io.wavfile
from scipy._lib._tmpdirs import tempdir
import scipy.sparse
def test_wavfile_read(self):
    path = Path(__file__).parent / 'data/test-8000Hz-le-2ch-1byteu.wav'
    scipy.io.wavfile.read(path)