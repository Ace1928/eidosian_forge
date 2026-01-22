from __future__ import annotations
import typing
import warnings
from pprint import pformat
from urwid.canvas import CanvasError, TextCanvas
from urwid.display.escape import SAFE_ASCII_DEC_SPECIAL_RE
from urwid.util import apply_target_encoding, str_util
class Sextant2x2Font(Font):
    name = 'Sextant 2x2'
    height = 2
    data = '\n..,,%%00112233445566778899\n    🬁🬖▐🬨🬇▌🬁🬗🬠🬸🬦▐▐🬒▐🬭🬁🬙▐🬸▐🬸\n🬇 🬇🬀🬁🬇🬉🬍 🬄🬉🬋🬇🬍🬁🬊🬇🬅🬉🬍 🬄🬉🬍 🬉\n'