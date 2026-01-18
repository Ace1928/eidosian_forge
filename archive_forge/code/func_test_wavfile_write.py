from pathlib import Path
import numpy as np
import scipy.io
import scipy.io.wavfile
from scipy._lib._tmpdirs import tempdir
import scipy.sparse
def test_wavfile_write(self):
    input_path = Path(__file__).parent / 'data/test-8000Hz-le-2ch-1byteu.wav'
    rate, data = scipy.io.wavfile.read(str(input_path))
    with tempdir() as temp_dir:
        output_path = Path(temp_dir) / input_path.name
        scipy.io.wavfile.write(output_path, rate, data)