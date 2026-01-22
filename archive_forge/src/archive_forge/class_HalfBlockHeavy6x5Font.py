from __future__ import annotations
import typing
import warnings
from pprint import pformat
from urwid.canvas import CanvasError, TextCanvas
from urwid.display.escape import SAFE_ASCII_DEC_SPECIAL_RE
from urwid.util import apply_target_encoding, str_util
class HalfBlockHeavy6x5Font(Font):
    name = 'Half Block Heavy 6x5'
    height = 5
    data = '\n000000111111222222333333444444555555666666777777888888999999  ..::////\n▄███▄  ▐█▌  ▄███▄ ▄███▄    █▌ █████ ▄███▄ █████ ▄███▄ ▄███▄         █▌\n█▌ ▐█  ▀█▌  ▀  ▐█ ▀  ▐█ █▌ █▌ █▌    █▌       █▌ █▌ ▐█ █▌ ▐█     █▌ ▐█\n█▌ ▐█   █▌    ▄█▀   ██▌ █████ ████▄ ████▄   ▐█  ▐███▌ ▀████        █▌\n█▌ ▐█   █▌  ▄█▀   ▄  ▐█    █▌    ▐█ █▌ ▐█   █▌  █▌ ▐█    ▐█     █▌▐█\n▀███▀  ███▌ █████ ▀███▀    █▌ ████▀ ▀███▀  ▐█   ▀███▀ ▀███▀   █▌  █▌\n'