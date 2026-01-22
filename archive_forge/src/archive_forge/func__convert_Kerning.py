from __future__ import annotations
import os
import shutil
from typing import (
import fs.base
import fs.tempfs
from attrs import define, field
from fontTools.ufoLib import UFOFileStructure, UFOReader, UFOWriter
from ufoLib2.constants import DEFAULT_LAYER_NAME
from ufoLib2.objects.dataSet import DataSet
from ufoLib2.objects.features import Features
from ufoLib2.objects.glyph import Glyph
from ufoLib2.objects.guideline import Guideline
from ufoLib2.objects.imageSet import ImageSet
from ufoLib2.objects.info import Info
from ufoLib2.objects.kerning import Kerning, KerningPair
from ufoLib2.objects.layer import Layer
from ufoLib2.objects.layerSet import LayerSet
from ufoLib2.objects.lib import (
from ufoLib2.objects.misc import (
from ufoLib2.serde import serde
from ufoLib2.typing import HasIdentifier, PathLike, T
def _convert_Kerning(value: Mapping[KerningPair, float]) -> Kerning:
    return value if isinstance(value, Kerning) else Kerning(value)