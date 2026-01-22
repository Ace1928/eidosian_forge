from . import matrix
def _compute_num_generators(choose_generators_info):
    """
    Compute the number of generators.
    """
    return max([max(info['generators']) for info in choose_generators_info])