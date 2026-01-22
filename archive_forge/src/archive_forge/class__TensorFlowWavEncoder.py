import numpy as np
from tensorboard.util import op_evaluator
class _TensorFlowWavEncoder(op_evaluator.PersistentOpEvaluator):
    """Encode an audio clip to WAV.

    This function is thread-safe and exhibits good parallel performance.

    Arguments:
      audio: A numpy array of shape `[samples, channels]`.
      samples_per_second: A positive `int`, in Hz.

    Returns:
      A bytestring with WAV-encoded data.
    """

    def __init__(self):
        super().__init__()
        self._audio_placeholder = None
        self._samples_per_second_placeholder = None
        self._encode_op = None

    def initialize_graph(self):
        import tensorflow.compat.v1 as tf
        self._audio_placeholder = tf.placeholder(dtype=tf.float32, name='image_to_encode')
        self._samples_per_second_placeholder = tf.placeholder(dtype=tf.int32, name='samples_per_second')
        self._encode_op = tf.audio.encode_wav(self._audio_placeholder, sample_rate=self._samples_per_second_placeholder)

    def run(self, audio, samples_per_second):
        if not isinstance(audio, np.ndarray):
            raise ValueError("'audio' must be a numpy array: %r" % audio)
        if not isinstance(samples_per_second, int):
            raise ValueError("'samples_per_second' must be an int: %r" % samples_per_second)
        feed_dict = {self._audio_placeholder: audio, self._samples_per_second_placeholder: samples_per_second}
        return self._encode_op.eval(feed_dict=feed_dict)