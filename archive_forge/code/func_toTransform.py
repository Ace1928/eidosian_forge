import math
from typing import NamedTuple
from dataclasses import dataclass
def toTransform(self):
    """Return the Transform() equivalent of this transformation.

        :Example:
                >>> DecomposedTransform(scaleX=2, scaleY=2).toTransform()
                <Transform [2 0 0 2 0 0]>
                >>>
        """
    t = Transform()
    t = t.translate(self.translateX + self.tCenterX, self.translateY + self.tCenterY)
    t = t.rotate(math.radians(self.rotation))
    t = t.scale(self.scaleX, self.scaleY)
    t = t.skew(math.radians(self.skewX), math.radians(self.skewY))
    t = t.translate(-self.tCenterX, -self.tCenterY)
    return t