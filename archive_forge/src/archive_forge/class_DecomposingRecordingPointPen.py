from fontTools.pens.basePen import AbstractPen, DecomposingPen
from fontTools.pens.pointPen import AbstractPointPen, DecomposingPointPen
class DecomposingRecordingPointPen(DecomposingPointPen, RecordingPointPen):
    """Same as RecordingPointPen, except that it doesn't keep components
    as references, but draws them decomposed as regular contours.

    The constructor takes a required 'glyphSet' positional argument,
    a dictionary of pointPen-drawable glyph objects (i.e. with a 'drawPoints' method)
    keyed by thir name; other arguments are forwarded to the DecomposingPointPen's
    constructor::

    >>> from pprint import pprint
    >>> class SimpleGlyph(object):
    ...     def drawPoints(self, pen):
    ...         pen.beginPath()
    ...         pen.addPoint((0, 0), "line")
    ...         pen.addPoint((1, 1))
    ...         pen.addPoint((2, 2))
    ...         pen.addPoint((3, 3), "curve")
    ...         pen.endPath()
    >>> class CompositeGlyph(object):
    ...     def drawPoints(self, pen):
    ...         pen.addComponent('a', (1, 0, 0, 1, -1, 1))
    >>> class MissingComponent(object):
    ...     def drawPoints(self, pen):
    ...         pen.addComponent('foobar', (1, 0, 0, 1, 0, 0))
    >>> class FlippedComponent(object):
    ...     def drawPoints(self, pen):
    ...         pen.addComponent('a', (-1, 0, 0, 1, 0, 0))
    >>> glyphSet = {
    ...    'a': SimpleGlyph(),
    ...    'b': CompositeGlyph(),
    ...    'c': MissingComponent(),
    ...    'd': FlippedComponent(),
    ... }
    >>> for name, glyph in sorted(glyphSet.items()):
    ...     pen = DecomposingRecordingPointPen(glyphSet)
    ...     try:
    ...         glyph.drawPoints(pen)
    ...     except pen.MissingComponentError:
    ...         pass
    ...     pprint({name: pen.value})
    {'a': [('beginPath', (), {}),
           ('addPoint', ((0, 0), 'line', False, None), {}),
           ('addPoint', ((1, 1), None, False, None), {}),
           ('addPoint', ((2, 2), None, False, None), {}),
           ('addPoint', ((3, 3), 'curve', False, None), {}),
           ('endPath', (), {})]}
    {'b': [('beginPath', (), {}),
           ('addPoint', ((-1, 1), 'line', False, None), {}),
           ('addPoint', ((0, 2), None, False, None), {}),
           ('addPoint', ((1, 3), None, False, None), {}),
           ('addPoint', ((2, 4), 'curve', False, None), {}),
           ('endPath', (), {})]}
    {'c': []}
    {'d': [('beginPath', (), {}),
           ('addPoint', ((0, 0), 'line', False, None), {}),
           ('addPoint', ((-1, 1), None, False, None), {}),
           ('addPoint', ((-2, 2), None, False, None), {}),
           ('addPoint', ((-3, 3), 'curve', False, None), {}),
           ('endPath', (), {})]}
    >>> for name, glyph in sorted(glyphSet.items()):
    ...     pen = DecomposingRecordingPointPen(
    ...         glyphSet, skipMissingComponents=True, reverseFlipped=True,
    ...     )
    ...     glyph.drawPoints(pen)
    ...     pprint({name: pen.value})
    {'a': [('beginPath', (), {}),
           ('addPoint', ((0, 0), 'line', False, None), {}),
           ('addPoint', ((1, 1), None, False, None), {}),
           ('addPoint', ((2, 2), None, False, None), {}),
           ('addPoint', ((3, 3), 'curve', False, None), {}),
           ('endPath', (), {})]}
    {'b': [('beginPath', (), {}),
           ('addPoint', ((-1, 1), 'line', False, None), {}),
           ('addPoint', ((0, 2), None, False, None), {}),
           ('addPoint', ((1, 3), None, False, None), {}),
           ('addPoint', ((2, 4), 'curve', False, None), {}),
           ('endPath', (), {})]}
    {'c': []}
    {'d': [('beginPath', (), {}),
           ('addPoint', ((0, 0), 'curve', False, None), {}),
           ('addPoint', ((-3, 3), 'line', False, None), {}),
           ('addPoint', ((-2, 2), None, False, None), {}),
           ('addPoint', ((-1, 1), None, False, None), {}),
           ('endPath', (), {})]}
    """
    skipMissingComponents = False