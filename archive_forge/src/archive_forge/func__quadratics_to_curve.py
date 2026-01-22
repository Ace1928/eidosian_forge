from fontTools.qu2cu import quadratic_to_curves
from fontTools.pens.filterPen import ContourFilterPen
from fontTools.pens.reverseContourPen import ReverseContourPen
import math
def _quadratics_to_curve(self, q):
    curves = quadratic_to_curves(q, self.max_err, all_cubic=self.all_cubic)
    if self.stats is not None:
        for curve in curves:
            n = str(len(curve) - 2)
            self.stats[n] = self.stats.get(n, 0) + 1
    for curve in curves:
        if len(curve) == 4:
            yield ('curveTo', curve[1:])
        else:
            yield ('qCurveTo', curve[1:])