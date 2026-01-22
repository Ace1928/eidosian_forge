import warnings
import numpy as np
import param
from packaging.version import Version
from param import _is_number
from ..core import (
from ..core.data import ArrayInterface, DictInterface, PandasInterface, default_datatype
from ..core.data.util import dask_array_module
from ..core.util import (
from ..element.chart import Histogram, Scatter
from ..element.path import Contours, Polygons
from ..element.raster import RGB, Image
from ..element.util import categorical_aggregate2d  # noqa (API import)
from ..streams import RangeXY
from ..util.locator import MaxNLocator
class chain(Operation):
    """
    Defining an Operation chain is an easy way to define a new
    Operation from a series of existing ones. The argument is a
    list of Operation (or Operation instances) that are
    called in sequence to generate the returned element.

    chain(operations=[gradient, threshold.instance(level=2)])

    This operation can accept an Image instance and would first
    compute the gradient before thresholding the result at a level of
    2.0.

    Instances are only required when arguments need to be passed to
    individual operations so the resulting object is a function over a
    single argument.
    """
    output_type = param.Parameter(default=Image, doc='\n        The output type of the chain operation. Must be supplied if\n        the chain is to be used as a channel operation.')
    group = param.String(default='', doc='\n        The group assigned to the result after having applied the chain.\n        Defaults to the group produced by the last operation in the chain')
    operations = param.List(default=[], item_type=Operation, doc='\n       A list of Operations (or Operation instances)\n       that are applied on the input from left to right.')

    def _process(self, view, key=None):
        processed = view
        for operation in self.p.operations:
            processed = operation.process_element(processed, key, input_ranges=self.p.input_ranges)
        if not self.p.group:
            return processed
        else:
            return processed.clone(group=self.p.group)

    def find(self, operation, skip_nonlinked=True):
        """
        Returns the first found occurrence of an operation while
        performing a backward traversal of the chain pipeline.
        """
        found = None
        for op in self.operations[::-1]:
            if isinstance(op, operation):
                found = op
                break
            if not op.link_inputs and skip_nonlinked:
                break
        return found