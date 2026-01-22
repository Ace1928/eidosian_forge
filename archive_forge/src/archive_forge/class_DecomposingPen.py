from typing import Tuple, Dict
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.transform import DecomposedTransform, Identity
class DecomposingPen(LoggingPen):
    """Implements a 'addComponent' method that decomposes components
    (i.e. draws them onto self as simple contours).
    It can also be used as a mixin class (e.g. see ContourRecordingPen).

    You must override moveTo, lineTo, curveTo and qCurveTo. You may
    additionally override closePath, endPath and addComponent.

    By default a warning message is logged when a base glyph is missing;
    set the class variable ``skipMissingComponents`` to False if you want
    all instances of a sub-class to raise a :class:`MissingComponentError`
    exception by default.
    """
    skipMissingComponents = True
    MissingComponentError = MissingComponentError

    def __init__(self, glyphSet, *args, skipMissingComponents=None, reverseFlipped=False, **kwargs):
        """Takes a 'glyphSet' argument (dict), in which the glyphs that are referenced
        as components are looked up by their name.

        If the optional 'reverseFlipped' argument is True, components whose transformation
        matrix has a negative determinant will be decomposed with a reversed path direction
        to compensate for the flip.

        The optional 'skipMissingComponents' argument can be set to True/False to
        override the homonymous class attribute for a given pen instance.
        """
        super(DecomposingPen, self).__init__(*args, **kwargs)
        self.glyphSet = glyphSet
        self.skipMissingComponents = self.__class__.skipMissingComponents if skipMissingComponents is None else skipMissingComponents
        self.reverseFlipped = reverseFlipped

    def addComponent(self, glyphName, transformation):
        """Transform the points of the base glyph and draw it onto self."""
        from fontTools.pens.transformPen import TransformPen
        try:
            glyph = self.glyphSet[glyphName]
        except KeyError:
            if not self.skipMissingComponents:
                raise MissingComponentError(glyphName)
            self.log.warning("glyph '%s' is missing from glyphSet; skipped" % glyphName)
        else:
            pen = self
            if transformation != Identity:
                pen = TransformPen(pen, transformation)
            if self.reverseFlipped:
                a, b, c, d = transformation[:4]
                det = a * d - b * c
                if det < 0:
                    from fontTools.pens.reverseContourPen import ReverseContourPen
                    pen = ReverseContourPen(pen)
            glyph.draw(pen)

    def addVarComponent(self, glyphName, transformation, location):
        raise AttributeError