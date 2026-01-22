import subprocess
from tempfile import NamedTemporaryFile
from .utils import get_player_name, make_chunks
def _play_with_pyaudio(seg):
    import pyaudio
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(seg.sample_width), channels=seg.channels, rate=seg.frame_rate, output=True)
    try:
        for chunk in make_chunks(seg, 500):
            stream.write(chunk._data)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()