from __future__ import absolute_import, division, print_function
import numpy as np
import mido
@property
def time_signatures(self):
    """
        Time signatures of the MIDI file.

        Returns
        -------
        time_signatures : numpy array
            Array with time signatures (time, numerator, denominator).

        Notes
        -----
        The time will be given in the unit set by `unit`.

        """
    signatures = []
    for msg in self:
        if msg.type == 'time_signature':
            signatures.append((msg.time, msg.numerator, msg.denominator))
    if not signatures or signatures[0][0] > 0:
        signatures.insert(0, (0, DEFAULT_TIME_SIGNATURE[0], DEFAULT_TIME_SIGNATURE[1]))
    return np.asarray(signatures, dtype=np.float)