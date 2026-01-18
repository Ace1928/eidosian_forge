import numpy as np
import os
import pkg_resources
from .containers import PitchBend
from .utilities import pitch_bend_to_semitones, note_number_to_hz
Synthesize using fluidsynth.

        Parameters
        ----------
        fs : int
            Sampling rate to synthesize.
        sf2_path : str
            Path to a .sf2 file.
            Default ``None``, which uses the TimGM6mb.sf2 file included with
            ``pretty_midi``.

        Returns
        -------
        synthesized : np.ndarray
            Waveform of the MIDI data, synthesized at ``fs``.

        