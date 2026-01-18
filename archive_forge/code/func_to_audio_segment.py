import math
import array
import itertools
import random
from .audio_segment import AudioSegment
from .utils import (
def to_audio_segment(self, duration=1000.0, volume=0.0):
    """
        Duration in milliseconds
            (default: 1 second)
        Volume in DB relative to maximum amplitude
            (default 0.0 dBFS, which is the maximum value)
        """
    minval, maxval = get_min_max_value(self.bit_depth)
    sample_width = get_frame_width(self.bit_depth)
    array_type = get_array_type(self.bit_depth)
    gain = db_to_float(volume)
    sample_count = int(self.sample_rate * (duration / 1000.0))
    sample_data = (int(val * maxval * gain) for val in self.generate())
    sample_data = itertools.islice(sample_data, 0, sample_count)
    data = array.array(array_type, sample_data)
    try:
        data = data.tobytes()
    except:
        data = data.tostring()
    return AudioSegment(data=data, metadata={'channels': 1, 'sample_width': sample_width, 'frame_rate': self.sample_rate, 'frame_width': sample_width})