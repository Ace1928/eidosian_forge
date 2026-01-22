from __future__ import annotations
from typing import (
from attrs import define, field
from fontTools.ufoLib.glifLib import GlyphSet
from ufoLib2.constants import DEFAULT_LAYER_NAME
from ufoLib2.objects.glyph import Glyph
from ufoLib2.objects.lib import (
from ufoLib2.objects.misc import (
from ufoLib2.serde import serde
from ufoLib2.typing import T
def addGlyph(self, glyph: Glyph) -> None:
    """Appends glyph object to the this layer unless its name is already
        taken."""
    self.insertGlyph(glyph, overwrite=False, copy=False)