import copy
import random
import numpy
from rdkit.DataStructs.VectCollection import VectCollection
from rdkit.ML import InfoTheory
from rdkit.ML.DecTree import SigTree

    **Arguments**

      - examples: the examples to be classified.  Each example
        should be a sequence at least three entries long, with
        entry 0 being a label, entry 1 a BitVector and entry -1
        an activity value

      - nPossibleRes: the number of result codes possible

      - ensemble: (optional) if this argument is provided, it
        should be a sequence which is used to limit the bits
        which are actually considered as potential descriptors.
        The default is None (use all bits).

      - random: (optional) If this argument is nonzero, it
        specifies the number of bits to be randomly selected
        for consideration at this node (i.e. this toggles the
        growth of Random Trees).
        The default is 0 (no random descriptor selection)

      - metric: (optional) This is an _InfoTheory.InfoType_ and
        sets the metric used to rank the bits.
        The default is _InfoTheory.InfoType.BIASENTROPY_

      - biasList: (optional) If provided, this provides a bias
        list for the bit ranker.
        See the _InfoTheory.InfoBitRanker_ docs for an explanation
        of bias.
        The default value is [1], which biases towards actives.

      - maxDepth: (optional) the maximum depth to which the tree
                   will be grown
        The default is -1 (no depth limit).

      - useCMIM: (optional) if this is >0, the CMIM algorithm
          (conditional mutual information maximization) will be
          used to select the descriptors used to build the trees.
          The value of the variable should be set to the number
          of descriptors to be used.  This option and the
          ensemble option are mutually exclusive (CMIM will not be
          used if the ensemble is set), but it happily coexsts
          with the random argument (to only consider random subsets
          of the top N CMIM bits)
        The default is 0 (do not use CMIM)

      - depth: (optional) the current depth in the tree
        This is used in the recursion and should not be set
        by the client.

    **Returns**

     a SigTree.SigTreeNode with the root of the decision tree

  