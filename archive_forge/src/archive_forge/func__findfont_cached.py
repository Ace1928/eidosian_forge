from base64 import b64encode
from collections import namedtuple
import copy
import dataclasses
from functools import lru_cache
from io import BytesIO
import json
import logging
from numbers import Number
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from typing import Union
import matplotlib as mpl
from matplotlib import _api, _afm, cbook, ft2font
from matplotlib._fontconfig_pattern import (
from matplotlib.rcsetup import _validators
@lru_cache(1024)
def _findfont_cached(self, prop, fontext, directory, fallback_to_default, rebuild_if_missing, rc_params):
    prop = FontProperties._from_any(prop)
    fname = prop.get_file()
    if fname is not None:
        return fname
    if fontext == 'afm':
        fontlist = self.afmlist
    else:
        fontlist = self.ttflist
    best_score = 1e+64
    best_font = None
    _log.debug('findfont: Matching %s.', prop)
    for font in fontlist:
        if directory is not None and Path(directory) not in Path(font.fname).parents:
            continue
        score = self.score_family(prop.get_family(), font.name) * 10 + self.score_style(prop.get_style(), font.style) + self.score_variant(prop.get_variant(), font.variant) + self.score_weight(prop.get_weight(), font.weight) + self.score_stretch(prop.get_stretch(), font.stretch) + self.score_size(prop.get_size(), font.size)
        _log.debug('findfont: score(%s) = %s', font, score)
        if score < best_score:
            best_score = score
            best_font = font
        if score == 0:
            break
    if best_font is None or best_score >= 10.0:
        if fallback_to_default:
            _log.warning('findfont: Font family %s not found. Falling back to %s.', prop.get_family(), self.defaultFamily[fontext])
            for family in map(str.lower, prop.get_family()):
                if family in font_family_aliases:
                    _log.warning('findfont: Generic family %r not found because none of the following families were found: %s', family, ', '.join(self._expand_aliases(family)))
            default_prop = prop.copy()
            default_prop.set_family(self.defaultFamily[fontext])
            return self.findfont(default_prop, fontext, directory, fallback_to_default=False)
        else:
            return _ExceptionProxy(ValueError, f'Failed to find font {prop}, and fallback to the default font was disabled')
    else:
        _log.debug('findfont: Matching %s to %s (%r) with score of %f.', prop, best_font.name, best_font.fname, best_score)
        result = best_font.fname
    if not os.path.isfile(result):
        if rebuild_if_missing:
            _log.info('findfont: Found a missing font file.  Rebuilding cache.')
            new_fm = _load_fontmanager(try_read_cache=False)
            vars(self).update(vars(new_fm))
            return self.findfont(prop, fontext, directory, rebuild_if_missing=False)
        else:
            return _ExceptionProxy(ValueError, 'No valid font could be found')
    return _cached_realpath(result)