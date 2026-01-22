from fontTools.misc.roundTools import noRound
from .errors import VariationModelError
class VariationModel(object):
    """Locations must have the base master at the origin (ie. 0).

    If the extrapolate argument is set to True, then values are extrapolated
    outside the axis range.

      >>> from pprint import pprint
      >>> locations = [       {'wght':100},       {'wght':-100},       {'wght':-180},       {'wdth':+.3},       {'wght':+120,'wdth':.3},       {'wght':+120,'wdth':.2},       {},       {'wght':+180,'wdth':.3},       {'wght':+180},       ]
      >>> model = VariationModel(locations, axisOrder=['wght'])
      >>> pprint(model.locations)
      [{},
       {'wght': -100},
       {'wght': -180},
       {'wght': 100},
       {'wght': 180},
       {'wdth': 0.3},
       {'wdth': 0.3, 'wght': 180},
       {'wdth': 0.3, 'wght': 120},
       {'wdth': 0.2, 'wght': 120}]
      >>> pprint(model.deltaWeights)
      [{},
       {0: 1.0},
       {0: 1.0},
       {0: 1.0},
       {0: 1.0},
       {0: 1.0},
       {0: 1.0, 4: 1.0, 5: 1.0},
       {0: 1.0, 3: 0.75, 4: 0.25, 5: 1.0, 6: 0.6666666666666666},
       {0: 1.0,
        3: 0.75,
        4: 0.25,
        5: 0.6666666666666667,
        6: 0.4444444444444445,
        7: 0.6666666666666667}]
    """

    def __init__(self, locations, axisOrder=None, extrapolate=False):
        if len(set((tuple(sorted(l.items())) for l in locations))) != len(locations):
            raise VariationModelError('Locations must be unique.')
        self.origLocations = locations
        self.axisOrder = axisOrder if axisOrder is not None else []
        self.extrapolate = extrapolate
        self.axisRanges = self.computeAxisRanges(locations) if extrapolate else None
        locations = [{k: v for k, v in loc.items() if v != 0.0} for loc in locations]
        keyFunc = self.getMasterLocationsSortKeyFunc(locations, axisOrder=self.axisOrder)
        self.locations = sorted(locations, key=keyFunc)
        self.mapping = [self.locations.index(l) for l in locations]
        self.reverseMapping = [locations.index(l) for l in self.locations]
        self._computeMasterSupports()
        self._subModels = {}

    def getSubModel(self, items):
        """Return a sub-model and the items that are not None.

        The sub-model is necessary for working with the subset
        of items when some are None.

        The sub-model is cached."""
        if None not in items:
            return (self, items)
        key = tuple((v is not None for v in items))
        subModel = self._subModels.get(key)
        if subModel is None:
            subModel = VariationModel(subList(key, self.origLocations), self.axisOrder)
            self._subModels[key] = subModel
        return (subModel, subList(key, items))

    @staticmethod
    def computeAxisRanges(locations):
        axisRanges = {}
        allAxes = {axis for loc in locations for axis in loc.keys()}
        for loc in locations:
            for axis in allAxes:
                value = loc.get(axis, 0)
                axisMin, axisMax = axisRanges.get(axis, (value, value))
                axisRanges[axis] = (min(value, axisMin), max(value, axisMax))
        return axisRanges

    @staticmethod
    def getMasterLocationsSortKeyFunc(locations, axisOrder=[]):
        if {} not in locations:
            raise VariationModelError('Base master not found.')
        axisPoints = {}
        for loc in locations:
            if len(loc) != 1:
                continue
            axis = next(iter(loc))
            value = loc[axis]
            if axis not in axisPoints:
                axisPoints[axis] = {0.0}
            assert value not in axisPoints[axis], 'Value "%s" in axisPoints["%s"] -->  %s' % (value, axis, axisPoints)
            axisPoints[axis].add(value)

        def getKey(axisPoints, axisOrder):

            def sign(v):
                return -1 if v < 0 else +1 if v > 0 else 0

            def key(loc):
                rank = len(loc)
                onPointAxes = [axis for axis, value in loc.items() if axis in axisPoints and value in axisPoints[axis]]
                orderedAxes = [axis for axis in axisOrder if axis in loc]
                orderedAxes.extend([axis for axis in sorted(loc.keys()) if axis not in axisOrder])
                return (rank, -len(onPointAxes), tuple((axisOrder.index(axis) if axis in axisOrder else 65536 for axis in orderedAxes)), tuple(orderedAxes), tuple((sign(loc[axis]) for axis in orderedAxes)), tuple((abs(loc[axis]) for axis in orderedAxes)))
            return key
        ret = getKey(axisPoints, axisOrder)
        return ret

    def reorderMasters(self, master_list, mapping):
        new_list = [master_list[idx] for idx in mapping]
        self.origLocations = [self.origLocations[idx] for idx in mapping]
        locations = [{k: v for k, v in loc.items() if v != 0.0} for loc in self.origLocations]
        self.mapping = [self.locations.index(l) for l in locations]
        self.reverseMapping = [locations.index(l) for l in self.locations]
        self._subModels = {}
        return new_list

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

    def _locationsToRegions(self):
        locations = self.locations
        minV = {}
        maxV = {}
        for l in locations:
            for k, v in l.items():
                minV[k] = min(v, minV.get(k, v))
                maxV[k] = max(v, maxV.get(k, v))
        regions = []
        for loc in locations:
            region = {}
            for axis, locV in loc.items():
                if locV > 0:
                    region[axis] = (0, locV, maxV[axis])
                else:
                    region[axis] = (minV[axis], locV, 0)
            regions.append(region)
        return regions

    def _computeDeltaWeights(self):
        self.deltaWeights = []
        for i, loc in enumerate(self.locations):
            deltaWeight = {}
            for j, support in enumerate(self.supports[:i]):
                scalar = supportScalar(loc, support)
                if scalar:
                    deltaWeight[j] = scalar
            self.deltaWeights.append(deltaWeight)

    def getDeltas(self, masterValues, *, round=noRound):
        assert len(masterValues) == len(self.deltaWeights)
        mapping = self.reverseMapping
        out = []
        for i, weights in enumerate(self.deltaWeights):
            delta = masterValues[mapping[i]]
            for j, weight in weights.items():
                if weight == 1:
                    delta -= out[j]
                else:
                    delta -= out[j] * weight
            out.append(round(delta))
        return out

    def getDeltasAndSupports(self, items, *, round=noRound):
        model, items = self.getSubModel(items)
        return (model.getDeltas(items, round=round), model.supports)

    def getScalars(self, loc):
        """Return scalars for each delta, for the given location.
        If interpolating many master-values at the same location,
        this function allows speed up by fetching the scalars once
        and using them with interpolateFromMastersAndScalars()."""
        return [supportScalar(loc, support, extrapolate=self.extrapolate, axisRanges=self.axisRanges) for support in self.supports]

    def getMasterScalars(self, targetLocation):
        """Return multipliers for each master, for the given location.
        If interpolating many master-values at the same location,
        this function allows speed up by fetching the scalars once
        and using them with interpolateFromValuesAndScalars().

        Note that the scalars used in interpolateFromMastersAndScalars(),
        are *not* the same as the ones returned here. They are the result
        of getScalars()."""
        out = self.getScalars(targetLocation)
        for i, weights in reversed(list(enumerate(self.deltaWeights))):
            for j, weight in weights.items():
                out[j] -= out[i] * weight
        out = [out[self.mapping[i]] for i in range(len(out))]
        return out

    @staticmethod
    def interpolateFromValuesAndScalars(values, scalars):
        """Interpolate from values and scalars coefficients.

        If the values are master-values, then the scalars should be
        fetched from getMasterScalars().

        If the values are deltas, then the scalars should be fetched
        from getScalars(); in which case this is the same as
        interpolateFromDeltasAndScalars().
        """
        v = None
        assert len(values) == len(scalars)
        for value, scalar in zip(values, scalars):
            if not scalar:
                continue
            contribution = value * scalar
            if v is None:
                v = contribution
            else:
                v += contribution
        return v

    @staticmethod
    def interpolateFromDeltasAndScalars(deltas, scalars):
        """Interpolate from deltas and scalars fetched from getScalars()."""
        return VariationModel.interpolateFromValuesAndScalars(deltas, scalars)

    def interpolateFromDeltas(self, loc, deltas):
        """Interpolate from deltas, at location loc."""
        scalars = self.getScalars(loc)
        return self.interpolateFromDeltasAndScalars(deltas, scalars)

    def interpolateFromMasters(self, loc, masterValues, *, round=noRound):
        """Interpolate from master-values, at location loc."""
        scalars = self.getMasterScalars(loc)
        return self.interpolateFromValuesAndScalars(masterValues, scalars)

    def interpolateFromMastersAndScalars(self, masterValues, scalars, *, round=noRound):
        """Interpolate from master-values, and scalars fetched from
        getScalars(), which is useful when you want to interpolate
        multiple master-values with the same location."""
        deltas = self.getDeltas(masterValues, round=round)
        return self.interpolateFromDeltasAndScalars(deltas, scalars)