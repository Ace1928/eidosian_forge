from .polynomial import Polynomial
from .component import NonZeroDimensionalComponent
from ..pari import pari
def _numerical_solutions_recursion(polysAndVars, solutionDict):
    if polysAndVars == []:
        return [solutionDict]
    univariatePoly = _get_first([poly for poly in polysAndVars if poly.get_variable_if_univariate()])
    if univariatePoly is not None:
        variable = univariatePoly.get_variable_if_univariate()
        variableDicts = []
        for solution in univariatePoly.get_roots():
            newSolutionDict = solutionDict.copy()
            newSolutionDict[variable] = solution
            new_polys = [poly.substitute(variable, solution) for poly in _remove(polysAndVars, univariatePoly)]
            variableDicts += _numerical_solutions_recursion(new_polys, newSolutionDict)
        return variableDicts
    return [solutionDict]