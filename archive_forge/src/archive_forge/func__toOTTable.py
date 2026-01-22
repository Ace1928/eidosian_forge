from fontTools.misc.textTools import safeEval
from . import DefaultTable
def _toOTTable(self, ttFont):
    from . import otTables
    from fontTools.colorLib.builder import populateCOLRv0
    tableClass = getattr(otTables, self.tableTag)
    table = tableClass()
    table.Version = self.version
    populateCOLRv0(table, {baseGlyph: [(layer.name, layer.colorID) for layer in layers] for baseGlyph, layers in self.ColorLayers.items()}, glyphMap=ttFont.getReverseGlyphMap(rebuild=True))
    return table