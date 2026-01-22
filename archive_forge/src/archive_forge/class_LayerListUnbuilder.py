from fontTools.ttLib.tables import otTables as ot
from .table_builder import TableUnbuilder
class LayerListUnbuilder:

    def __init__(self, layers):
        self.layers = layers
        callbacks = {(ot.Paint, ot.PaintFormat.PaintColrLayers): self._unbuildPaintColrLayers}
        self.tableUnbuilder = TableUnbuilder(callbacks)

    def unbuildPaint(self, paint):
        assert isinstance(paint, ot.Paint)
        return self.tableUnbuilder.unbuild(paint)

    def _unbuildPaintColrLayers(self, source):
        assert source['Format'] == ot.PaintFormat.PaintColrLayers
        layers = list(_flatten_layers([self.unbuildPaint(childPaint) for childPaint in self.layers[source['FirstLayerIndex']:source['FirstLayerIndex'] + source['NumLayers']]]))
        if len(layers) == 1:
            return layers[0]
        return {'Format': source['Format'], 'Layers': layers}