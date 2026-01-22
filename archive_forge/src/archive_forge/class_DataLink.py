import weakref
from collections import defaultdict
import param
from ..core.util import dimension_sanitizer
class DataLink(Link):
    """
    DataLink defines a link in the data between two objects allowing
    them to be selected together. In order for a DataLink to be
    established the source and target data must be of the same length.
    """
    _requires_target = True