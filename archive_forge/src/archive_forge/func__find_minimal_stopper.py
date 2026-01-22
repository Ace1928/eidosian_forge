from ..sage_helper import _within_sage
from ..math_basics import correct_min, is_RealIntervalFieldElement
def _find_minimal_stopper(cusp_area_matrix, assigned_areas):
    return min(_find_potential_stoppers(cusp_area_matrix, assigned_areas))