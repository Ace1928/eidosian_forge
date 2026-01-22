import colorsys  # colour format conversions
from math import log, exp, floor, pi
import random  # for jitter values
class ColorSpiral:
    """Implement a spiral path through HSV colour space.

    This class provides functions for sampling points along a logarithmic
    spiral path through HSV colour space.

    The spiral is described by r = a * exp(b * t) where r is the distance
    from the axis of the HSV cylinder to the current point in the spiral,
    and t is the angle through which the spiral has turned to reach the
    current point. a and b are (positive, real) parameters that control the
    shape of the spiral.

     - a: the starting direction of the spiral
     - b: the number of revolutions about the axis made by the spiral

    We permit the spiral to move along the cylinder ('in V-space') between
    v_init and v_final, to give a gradation in V (essentially, brightness),
    along the path, where v_init, v_final are in [0,1].

    A brightness 'jitter' may also be provided as an absolute value in
    V-space, to aid in distinguishing consecutive colour points on the
    path.
    """

    def __init__(self, a=1, b=0.33, v_init=0.85, v_final=0.5, jitter=0.05):
        """Initialize a logarithmic spiral path through HSV colour space.

        Arguments:
         - a - Parameter a for the spiral, controls the initial spiral
           direction. a > 0
         - b - parameter b for the spiral, controls the rate at which the
           spiral revolves around the axis. b > 0
         - v_init - initial value of V (brightness) for the spiral.
           v_init in [0,1]
         - v_final - final value of V (brightness) for the spiral
           v_final in [0,1]
         - jitter - the degree of V (brightness) jitter to add to each
           selected colour. The amount of jitter will be selected
           from a uniform random distribution [-jitter, jitter],
           and V will be maintained in [0,1].

        """
        self.a = a
        self.b = b
        self.v_init = v_init
        self.v_final = v_final
        self.jitter = jitter

    def get_colors(self, k, offset=0.1):
        """Generate k different RBG colours evenly-space on the spiral.

        A generator returning the RGB colour space values for k
        evenly-spaced points along the defined spiral in HSV space.

        Arguments:
         - k - the number of points to return
         - offset - how far along the spiral path to start.

        """
        assert offset > 0 and offset < 1, 'offset must be in (0,1)'
        v_rate = (self._v_final - self._v_init) / k
        for n in range(1, k + 1):
            t = 1.0 / self._b * (log(n + k * offset) - log((1 + offset) * k * self._a))
            h = t
            while h < 0:
                h += 2 * pi
            h = h - floor(h / (2 * pi)) * pi
            h = h / (2 * pi)
            r = self._a * exp(self._b * t)
            if self._jitter:
                jitter = random.random() * 2 * self._jitter - self._jitter
            else:
                jitter = 0
            v = self._v_init + (n * v_rate + jitter)
            yield colorsys.hsv_to_rgb(h, r, max(0, min(v, 1)))

    def _get_a(self):
        return self._a

    def _set_a(self, value):
        self._a = max(0, value)

    def _get_b(self):
        return self._b

    def _set_b(self, value):
        self._b = max(0, value)

    def _get_v_init(self):
        return self._v_init

    def _set_v_init(self, value):
        self._v_init = max(0, min(1, value))

    def _get_v_final(self):
        return self._v_final

    def _set_v_final(self, value):
        self._v_final = max(0, min(1, value))

    def _get_jitter(self):
        return self._jitter

    def _set_jitter(self, value):
        self._jitter = max(0, min(1, value))
    a = property(_get_a, _set_a, doc='Parameter controlling initial spiral direction (a > 0)')
    b = property(_get_b, _set_b, doc='Parameter controlling rate spiral revolves around axis (b > 0)')
    v_init = property(_get_v_init, _set_v_init, doc='Initial value of V (brightness) for the spiral (range 0 to 1)')
    v_final = property(_get_v_final, _set_v_final, doc='Final value of V (brightness) for the spiral (range 0 to 1)')
    jitter = property(_get_jitter, _set_jitter, doc='Degree of V (brightness) jitter to add to each color (range 0 to 1)')