from fontTools.misc.roundTools import noRound
from .errors import VariationModelError
def _computeMasterSupports(self):
    self.supports = []
    regions = self._locationsToRegions()
    for i, region in enumerate(regions):
        locAxes = set(region.keys())
        for prev_region in regions[:i]:
            if set(prev_region.keys()) != locAxes:
                continue
            relevant = True
            for axis, (lower, peak, upper) in region.items():
                if not (prev_region[axis][1] == peak or lower < prev_region[axis][1] < upper):
                    relevant = False
                    break
            if not relevant:
                continue
            bestAxes = {}
            bestRatio = -1
            for axis in prev_region.keys():
                val = prev_region[axis][1]
                assert axis in region
                lower, locV, upper = region[axis]
                newLower, newUpper = (lower, upper)
                if val < locV:
                    newLower = val
                    ratio = (val - locV) / (lower - locV)
                elif locV < val:
                    newUpper = val
                    ratio = (val - locV) / (upper - locV)
                else:
                    continue
                if ratio > bestRatio:
                    bestAxes = {}
                    bestRatio = ratio
                if ratio == bestRatio:
                    bestAxes[axis] = (newLower, locV, newUpper)
            for axis, triple in bestAxes.items():
                region[axis] = triple
        self.supports.append(region)
    self._computeDeltaWeights()