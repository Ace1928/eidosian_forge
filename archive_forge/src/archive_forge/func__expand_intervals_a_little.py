from snappy import snap
from snappy.sage_helper import _within_sage, sage_method
@staticmethod
def _expand_intervals_a_little(shapes):
    """
        Make the intervals a tiny bit larger.
        """
    return shapes.apply_map(lambda z: z + (z - z) / 64)