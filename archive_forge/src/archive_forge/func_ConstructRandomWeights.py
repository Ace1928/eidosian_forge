import random
import numpy
from rdkit.ML.Neural import ActFuncs, NetNode
def ConstructRandomWeights(self, minWeight=-1, maxWeight=1):
    """initialize all the weights in the network to random numbers

      **Arguments**

        - minWeight: the minimum value a weight can take

        - maxWeight: the maximum value a weight can take

    """
    for node in self.nodeList:
        inputs = node.GetInputs()
        if inputs:
            weights = [random.uniform(minWeight, maxWeight) for _ in range(len(inputs))]
            node.SetWeights(weights)