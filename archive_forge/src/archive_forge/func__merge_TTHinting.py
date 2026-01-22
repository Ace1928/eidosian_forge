from typing import List
from fontTools.misc.vector import Vector
from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.fixedTools import floatToFixed as fl2fi
from fontTools.misc.textTools import Tag, tostr
from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables._f_v_a_r import Axis, NamedInstance
from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates, dropImpliedOnCurvePoints
from fontTools.ttLib.tables.ttProgram import Program
from fontTools.ttLib.tables.TupleVariation import TupleVariation
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import OTTableWriter
from fontTools.varLib import builder, models, varStore
from fontTools.varLib.merger import VariationMerger, COLRVariationMerger
from fontTools.varLib.mvar import MVAR_ENTRIES
from fontTools.varLib.iup import iup_delta_optimize
from fontTools.varLib.featureVars import addFeatureVariations
from fontTools.designspaceLib import DesignSpaceDocument, InstanceDescriptor
from fontTools.designspaceLib.split import splitInterpolable, splitVariableFonts
from fontTools.varLib.stat import buildVFStatTable
from fontTools.colorLib.builder import buildColrV1
from fontTools.colorLib.unbuilder import unbuildColrV1
from functools import partial
from collections import OrderedDict, defaultdict, namedtuple
import os.path
import logging
from copy import deepcopy
from pprint import pformat
from re import fullmatch
from .errors import VarLibError, VarLibValidationError
def _merge_TTHinting(font, masterModel, master_ttfs):
    log.info('Merging TT hinting')
    assert 'cvar' not in font
    for tag in ('fpgm', 'prep'):
        all_pgms = [m[tag].program for m in master_ttfs if tag in m]
        if not all_pgms:
            continue
        font_pgm = getattr(font.get(tag), 'program', None)
        if any((pgm != font_pgm for pgm in all_pgms)):
            log.warning('Masters have incompatible %s tables, hinting is discarded.' % tag)
            _remove_TTHinting(font)
            return
    font_glyf = font['glyf']
    master_glyfs = [m['glyf'] for m in master_ttfs]
    for name, glyph in font_glyf.glyphs.items():
        all_pgms = [getattr(glyf.get(name), 'program', None) for glyf in master_glyfs]
        if not any(all_pgms):
            continue
        glyph.expand(font_glyf)
        font_pgm = getattr(glyph, 'program', None)
        if any((pgm != font_pgm for pgm in all_pgms if pgm)):
            log.warning("Masters have incompatible glyph programs in glyph '%s', hinting is discarded." % name)
            _remove_TTHinting(font)
            return
    all_cvs = [Vector(m['cvt '].values) if 'cvt ' in m else None for m in master_ttfs]
    nonNone_cvs = models.nonNone(all_cvs)
    if not nonNone_cvs:
        return
    if not models.allEqual((len(c) for c in nonNone_cvs)):
        log.warning('Masters have incompatible cvt tables, hinting is discarded.')
        _remove_TTHinting(font)
        return
    variations = []
    deltas, supports = masterModel.getDeltasAndSupports(all_cvs, round=round)
    for i, (delta, support) in enumerate(zip(deltas[1:], supports[1:])):
        if all((v == 0 for v in delta)):
            continue
        var = TupleVariation(support, delta)
        variations.append(var)
    if variations:
        cvar = font['cvar'] = newTable('cvar')
        cvar.version = 1
        cvar.variations = variations