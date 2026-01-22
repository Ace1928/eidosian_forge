from __future__ import absolute_import, division, print_function
from functools import partial
import numpy as np
from ..io import SEGMENT_DTYPE
from ..processors import SequentialProcessor
class DeepChromaChordRecognitionProcessor(SequentialProcessor):
    """
    Recognise major and minor chords from deep chroma vectors [1]_ using a
    Conditional Random Field.

    Parameters
    ----------
    model : str
        File containing the CRF model. If None, use the model supplied with
        madmom.
    fps : float
        Frames per second. Must correspond to the fps of the incoming
        activations and the model.

    References
    ----------
    .. [1] Filip Korzeniowski and Gerhard Widmer,
           "Feature Learning for Chord Recognition: The Deep Chroma Extractor",
           Proceedings of the 17th International Society for Music Information
           Retrieval Conference (ISMIR), 2016.

    Examples
    --------
    To recognise chords in an audio file using the
    DeepChromaChordRecognitionProcessor you first need to create a
    madmom.audio.chroma.DeepChromaProcessor to extract the appropriate chroma
    vectors.

    >>> from madmom.audio.chroma import DeepChromaProcessor
    >>> dcp = DeepChromaProcessor()
    >>> dcp  # doctest: +ELLIPSIS
    <madmom.audio.chroma.DeepChromaProcessor object at ...>

    Then, create the DeepChromaChordRecognitionProcessor to decode a chord
    sequence from the extracted chromas:

    >>> decode = DeepChromaChordRecognitionProcessor()
    >>> decode  # doctest: +ELLIPSIS
    <madmom.features.chords.DeepChromaChordRecognitionProcessor object at ...>

    To transcribe the chords, you can either manually call the processors
    one after another,

    >>> chroma = dcp('tests/data/audio/sample2.wav')
    >>> decode(chroma)
    ... # doctest: +NORMALIZE_WHITESPACE +NORMALIZE_ARRAYS
    array([(0. , 1.6, 'F:maj'), (1.6, 2.5, 'A:maj'), (2.5, 4.1, 'D:maj')],
          dtype=[('start', '<f8'), ('end', '<f8'), ('label', 'O')])

    or create a `SequentialProcessor` that connects them:

    >>> from madmom.processors import SequentialProcessor
    >>> chordrec = SequentialProcessor([dcp, decode])
    >>> chordrec('tests/data/audio/sample2.wav')
    ... # doctest: +NORMALIZE_WHITESPACE +NORMALIZE_ARRAYS
    array([(0. , 1.6, 'F:maj'), (1.6, 2.5, 'A:maj'), (2.5, 4.1, 'D:maj')],
          dtype=[('start', '<f8'), ('end', '<f8'), ('label', 'O')])
    """

    def __init__(self, model=None, fps=10, **kwargs):
        from ..ml.crf import ConditionalRandomField
        from ..models import CHORDS_DCCRF
        crf = ConditionalRandomField.load(model or CHORDS_DCCRF[0])
        lbl = partial(majmin_targets_to_chord_labels, fps=fps)
        super(DeepChromaChordRecognitionProcessor, self).__init__((crf, lbl))