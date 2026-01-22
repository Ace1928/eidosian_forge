import copy
import re
import numpy as np
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.lib import debug_data
class HighlightOptions(object):
    """Options for highlighting elements of a tensor."""

    def __init__(self, criterion, description=None, font_attr=DEFAULT_TENSOR_ELEMENT_HIGHLIGHT_FONT_ATTR):
        """Constructor of HighlightOptions.

    Args:
      criterion: (callable) A callable of the following signature:
        def to_highlight(X):
          # Args:
          #   X: The tensor to highlight elements in.
          #
          # Returns:
          #   (boolean ndarray) A boolean ndarray of the same shape as X
          #   indicating which elements are to be highlighted (iff True).
        This callable will be used as the argument of np.argwhere() to
        determine which elements of the tensor are to be highlighted.
      description: (str) Description of the highlight criterion embodied by
        criterion.
      font_attr: (str) Font attribute to be applied to the
        highlighted elements.

    """
        self.criterion = criterion
        self.description = description
        self.font_attr = font_attr