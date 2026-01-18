from fontTools.varLib.models import VariationModel, normalizeValue, piecewiseLinearMap
def value_at_location(self, location, model_cache=None, avar=None):
    loc = location
    if loc in self.values.keys():
        return self.values[loc]
    values = list(self.values.values())
    return self.model(model_cache, avar).interpolateFromMasters(loc, values)