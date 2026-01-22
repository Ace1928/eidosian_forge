import random
import numpy
from rdkit.ML.Neural import ActFuncs, NetNode
 Constructor

      This constructs and initializes the network based upon the specified
      node counts.

      A fully connected network with random weights is constructed.

      **Arguments**

        - nodeCounts: a list containing the number of nodes to be in each layer.
           the ordering is:
            (nInput,nHidden1,nHidden2, ... , nHiddenN, nOutput)

        - nodeConnections: I don't know why this is here, but it's optional.  ;-)

        - actFunc: the activation function to be used here.  Must support the API
            of _ActFuncs.ActFunc_.

        - actFuncParms: a tuple of extra arguments to be passed to the activation function
            constructor.

        - weightBounds:  a float which provides the boundary on the random initial weights

    