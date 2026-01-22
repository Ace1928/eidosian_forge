import random
import numpy
from rdkit.ML.Neural import ActFuncs, NetNode
def ConstructNodes(self, nodeCounts, actFunc, actFuncParms):
    """ build an unconnected network and set node counts

      **Arguments**

        - nodeCounts: a list containing the number of nodes to be in each layer.
           the ordering is:
            (nInput,nHidden1,nHidden2, ... , nHiddenN, nOutput)

    """
    self.nodeCounts = nodeCounts
    self.numInputNodes = nodeCounts[0]
    self.numOutputNodes = nodeCounts[-1]
    self.numHiddenLayers = len(nodeCounts) - 2
    self.numInHidden = [None] * self.numHiddenLayers
    for i in range(self.numHiddenLayers):
        self.numInHidden[i] = nodeCounts[i + 1]
    numNodes = sum(self.nodeCounts)
    self.nodeList = [None] * numNodes
    for i in range(numNodes):
        self.nodeList[i] = NetNode.NetNode(i, self.nodeList, actFunc=actFunc, actFuncParms=actFuncParms)
    self.layerIndices = [None] * len(nodeCounts)
    start = 0
    for i in range(len(nodeCounts)):
        end = start + nodeCounts[i]
        self.layerIndices[i] = list(range(start, end))
        start = end