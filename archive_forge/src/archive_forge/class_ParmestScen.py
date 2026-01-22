import pyomo.environ as pyo
class ParmestScen(object):
    """A little container for scenarios; the Args are the attributes.

    Args:
        name (str): name for reporting; might be ""
        ThetaVals (dict): ThetaVals[name]=val
        probability (float): probability of occurrence "near" these ThetaVals
    """

    def __init__(self, name, ThetaVals, probability):
        self.name = name
        assert isinstance(ThetaVals, dict)
        self.ThetaVals = ThetaVals
        self.probability = probability