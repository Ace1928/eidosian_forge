from fontTools.misc.fixedTools import (
from fontTools.varLib.models import normalizeValue, piecewiseLinearMap
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables.TupleVariation import TupleVariation
from fontTools.ttLib.tables import _g_l_y_f
from fontTools import varLib
from fontTools import subset  # noqa: F401
from fontTools.varLib import builder
from fontTools.varLib.mvar import MVAR_ENTRIES
from fontTools.varLib.merger import MutatorMerger
from fontTools.varLib.instancer import names
from .featureVars import instantiateFeatureVariations
from fontTools.misc.cliTools import makeOutputFileName
from fontTools.varLib.instancer import solver
import collections
import dataclasses
from contextlib import contextmanager
from copy import deepcopy
from enum import IntEnum
import logging
import os
import re
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union
import warnings
@contextmanager
def verticalMetricsKeptInSync(varfont):
    """Ensure hhea vertical metrics stay in sync with OS/2 ones after instancing.

    When applying MVAR deltas to the OS/2 table, if the ascender, descender and
    line gap change but they were the same as the respective hhea metrics in the
    original font, this context manager ensures that hhea metrcs also get updated
    accordingly.
    The MVAR spec only has tags for the OS/2 metrics, but it is common in fonts
    to have the hhea metrics be equal to those for compat reasons.

    https://learn.microsoft.com/en-us/typography/opentype/spec/mvar
    https://googlefonts.github.io/gf-guide/metrics.html#7-hhea-and-typo-metrics-should-be-equal
    https://github.com/fonttools/fonttools/issues/3297
    """
    current_os2_vmetrics = [getattr(varfont['OS/2'], attr) for attr in ('sTypoAscender', 'sTypoDescender', 'sTypoLineGap')]
    metrics_are_synced = current_os2_vmetrics == [getattr(varfont['hhea'], attr) for attr in ('ascender', 'descender', 'lineGap')]
    yield metrics_are_synced
    if metrics_are_synced:
        new_os2_vmetrics = [getattr(varfont['OS/2'], attr) for attr in ('sTypoAscender', 'sTypoDescender', 'sTypoLineGap')]
        if current_os2_vmetrics != new_os2_vmetrics:
            for attr, value in zip(('ascender', 'descender', 'lineGap'), new_os2_vmetrics):
                setattr(varfont['hhea'], attr, value)