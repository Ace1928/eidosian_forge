from __future__ import annotations
import typing
import warnings
from pprint import pformat
from urwid.canvas import CanvasError, TextCanvas
from urwid.display.escape import SAFE_ASCII_DEC_SPECIAL_RE
from urwid.util import apply_target_encoding, str_util
class Sextant3x3Font(Font):
    name = 'Sextant 3x3'
    height = 3
    data = ("\n   !!!###$$$%%%&&&'''((()))***+++,,,---...///\n    ▐ 🬞🬲🬲🬞🬍🬋🬉🬄🬖🬦🬧  🬉 🬞🬅 🬁🬢 🬞🬦🬞 🬦            🬖\n    🬉 🬇🬛🬛🬞🬰🬗🬞🬅🬭🬦🬈🬖   🬉🬏  🬘 🬇🬨🬈🬁🬨🬂 🬭 🬁🬂🬂 🬭 🬞🬅\n    🬁  🬀🬀 🬁   🬂 🬂🬁    🬁 🬁         🬅     🬂\n", '\n000111222333444555666777888999\n🬦🬂🬧🬞🬫 🬇🬂🬧🬁🬂🬧 🬞🬫▐🬂🬂🬞🬅🬀🬁🬂🬙🬦🬂🬧🬦🬂🬧\n▐🬁▐ ▐ 🬞🬅🬀 🬂🬧🬇🬌🬫🬁🬂🬧▐🬂🬧 🬔 🬦🬂🬧 🬂🬙\n 🬂🬀 🬁 🬁🬂🬂🬁🬂🬀  🬁🬁🬂🬀 🬂🬀 🬀  🬂🬀 🬂\n', '\n"""\n 🬄🬄\n\n\n', '\n:::;;;<<<===>>>???@@@\n 🬭  🬭  🬖🬀   🬁🬢 🬇🬂🬧🬦🬂🬧\n 🬰  🬰 🬁🬢 🬠🬰🬰 🬖🬀 🬇🬀▐🬉🬅\n 🬂  🬅   🬀   🬁   🬁  🬂🬀\n', '\nAAABBBCCCDDDEEEFFFGGGHHHIIIJJJKKKLLLMMMNNNOOOPPPQQQ\n🬞🬅🬢▐🬂🬧🬦🬂🬈▐🬂🬧▐🬂🬂▐🬂🬂🬦🬂🬈▐ ▐ 🬨🬀  ▐▐🬞🬅▐  ▐🬢🬫▐🬢▐🬦🬂🬧▐🬂🬧🬦🬂🬧\n▐🬋🬫▐🬂🬧▐ 🬞▐ ▐▐🬂 ▐🬂 ▐ 🬨▐🬂🬨 ▐ 🬞 ▐▐🬈🬏▐  ▐🬁▐▐ 🬨▐ ▐▐🬂🬀▐🬇🬘\n🬁 🬁🬁🬂🬀 🬂🬀🬁🬂🬀🬁🬂🬂🬁   🬂🬂🬁 🬁 🬂🬀 🬂🬀🬁 🬁🬁🬂🬂🬁 🬁🬁 🬁 🬂🬀🬁   🬂🬁\n', '\nRRRSSSTTTUUUVVVWWWXXXYYYZZZ[[[]]]^^^___```\n▐🬂🬧🬦🬂🬈🬁🬨🬂▐ ▐▐ ▐▐ ▐🬉🬏🬘▐ ▐🬁🬂🬙 🬕🬀 🬂▌🬞🬅🬢    🬈🬏\n▐🬊🬐🬞🬂🬧 ▐ ▐ ▐🬉🬏🬘▐🬖🬷🬞🬅🬢 🬧🬀🬞🬅  ▌   ▌\n🬁 🬁 🬂🬀 🬁  🬂🬀 🬁 🬁 🬁🬁 🬁 🬁 🬁🬂🬂 🬂🬀 🬂🬀   🬂🬂🬂\n', '\n\\\\\\\n🬇🬏\n 🬁🬢\n\n')