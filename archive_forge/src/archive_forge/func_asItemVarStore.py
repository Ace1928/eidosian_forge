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
def asItemVarStore(self):
    regionOrder = [frozenset(axes.items()) for axes in self.regions]
    varDatas = []
    for variations, itemCount in zip(self.tupleVarData, self.itemCounts):
        if variations:
            assert len(variations[0].coordinates) == itemCount
            varRegionIndices = [regionOrder.index(frozenset(var.axes.items())) for var in variations]
            varDataItems = list(zip(*(var.coordinates for var in variations)))
            varDatas.append(builder.buildVarData(varRegionIndices, varDataItems, optimize=False))
        else:
            varDatas.append(builder.buildVarData([], [[] for _ in range(itemCount)]))
    regionList = builder.buildVarRegionList(self.regions, self.axisOrder)
    itemVarStore = builder.buildVarStore(regionList, varDatas)
    itemVarStore.prune_regions()
    return itemVarStore