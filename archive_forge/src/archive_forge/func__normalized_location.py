from fontTools.varLib.models import VariationModel, normalizeValue, piecewiseLinearMap
def _normalized_location(self, location):
    location = self.fix_location(location)
    normalized_location = {}
    for axtag in location.keys():
        if axtag not in self.axes_dict:
            raise ValueError('Unknown axis %s in %s' % (axtag, location))
        axis = self.axes_dict[axtag]
        normalized_location[axtag] = normalizeValue(location[axtag], (axis.minValue, axis.defaultValue, axis.maxValue))
    return Location(normalized_location)