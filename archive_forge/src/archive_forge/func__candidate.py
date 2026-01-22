import pickle
import base64
import zlib
from re import match as re_match
from collections import deque
from math import sqrt, pi, radians, acos, atan, atan2, pow, floor
from math import sin as math_sin, cos as math_cos
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.properties import ListProperty
from kivy.compat import PY2
from io import BytesIO
def _candidate(self, strokes, **kwargs):
    if isinstance(strokes, Candidate):
        return strokes
    if not isinstance(strokes, list) or not len(strokes) or (not isinstance(strokes[0], list)):
        raise MultistrokeError('recognize() needs strokes= list or Candidate')
    cand = Candidate(strokes)
    o_filter = kwargs.get('orientation_sensitive', None)
    if o_filter is False:
        cand.skip_bounded = True
    elif o_filter is True:
        cand.skip_invariant = True
    return cand