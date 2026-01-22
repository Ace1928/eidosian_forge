import math
from typing import NamedTuple
from dataclasses import dataclass
@dataclass
class DecomposedTransform:
    """The DecomposedTransform class implements a transformation with separate
    translate, rotation, scale, skew, and transformation-center components.
    """
    translateX: float = 0
    translateY: float = 0
    rotation: float = 0
    scaleX: float = 1
    scaleY: float = 1
    skewX: float = 0
    skewY: float = 0
    tCenterX: float = 0
    tCenterY: float = 0

    @classmethod
    def fromTransform(self, transform):
        a, b, c, d, x, y = transform
        sx = math.copysign(1, a)
        if sx < 0:
            a *= sx
            b *= sx
        delta = a * d - b * c
        rotation = 0
        scaleX = scaleY = 0
        skewX = skewY = 0
        if a != 0 or b != 0:
            r = math.sqrt(a * a + b * b)
            rotation = math.acos(a / r) if b >= 0 else -math.acos(a / r)
            scaleX, scaleY = (r, delta / r)
            skewX, skewY = (math.atan((a * c + b * d) / (r * r)), 0)
        elif c != 0 or d != 0:
            s = math.sqrt(c * c + d * d)
            rotation = math.pi / 2 - (math.acos(-c / s) if d >= 0 else -math.acos(c / s))
            scaleX, scaleY = (delta / s, s)
            skewX, skewY = (0, math.atan((a * c + b * d) / (s * s)))
        else:
            pass
        return DecomposedTransform(x, y, math.degrees(rotation), scaleX * sx, scaleY, math.degrees(skewX) * sx, math.degrees(skewY), 0, 0)

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