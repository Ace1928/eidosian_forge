import numpy
from rdkit.ML.Composite import Composite
 classifies the given example using the entire composite

      **Arguments**

       - example: the data to be classified

       - threshold:  if this is a number greater than zero, then a
          classification will only be returned if the confidence is
          above _threshold_.  Anything lower is returned as -1.

      **Returns**

        a (result,confidence) tuple

    