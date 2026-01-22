from __future__ import print_function
from .utilities import key_number_to_key_name
class Lyric(object):
    """Timestamped lyric text.

    Attributes
    ----------
    text : str
        The text of the lyric.
    time : float
        The time in seconds of the lyric.
    """

    def __init__(self, text, time):
        self.text = text
        self.time = time

    def __repr__(self):
        return 'Lyric(text="{}", time={})'.format(self.text.replace('"', '\\"'), self.time)

    def __str__(self):
        return '"{}" at {:.2f} seconds'.format(self.text, self.time)