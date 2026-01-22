import numpy
def StepUpdate(self, example, net, resVect=None):
    """ does a BackProp step based upon the example

      **Arguments**

        - example: a 2-tuple:
           1) a list of variable values values
           2) a list of result values (targets)

        - net: a _Network_ (or something supporting the same API)

        - resVect: if this is nonzero, then the network is not required to
          classify the _example_

      **Returns**

        the backprop error from _network_ **before the update**

      **Note**

        In case it wasn't blindingly obvious, the weights in _network_ are modified
        in the course of taking a backprop step.

    """
    totNumNodes = net.GetNumNodes()
    if self.oldDeltaW is None:
        self.oldDeltaW = numpy.zeros(totNumNodes, numpy.float64)
    outputNodeList = net.GetOutputNodeList()
    nOutput = len(outputNodeList)
    targetVect = numpy.array(example[-nOutput:], numpy.float64)
    trainVect = example[:-nOutput]
    if resVect is None:
        net.ClassifyExample(trainVect)
        resVect = net.GetLastOutputs()
    outputs = numpy.take(resVect, outputNodeList)
    errVect = targetVect - outputs
    delta = numpy.zeros(totNumNodes, numpy.float64)
    for i in range(len(outputNodeList)):
        idx = outputNodeList[i]
        node = net.GetNode(idx)
        delta[idx] = errVect[i] * node.actFunc.DerivFromVal(resVect[idx])
        inputs = node.GetInputs()
        weights = delta[idx] * node.GetWeights()
        for j in range(len(inputs)):
            idx2 = inputs[j]
            delta[idx2] = delta[idx2] + weights[j]
    for layer in range(net.GetNumHidden() - 1, -1, -1):
        nodesInLayer = net.GetHiddenLayerNodeList(layer)
        for idx in nodesInLayer:
            node = net.GetNode(idx)
            delta[idx] = delta[idx] * node.actFunc.DerivFromVal(resVect[idx])
            if layer != 0:
                inputs = node.GetInputs()
                weights = delta[idx] * node.GetWeights()
                for i in range(len(inputs)):
                    idx2 = inputs[i]
                    delta[idx2] = delta[idx2] + weights[i]
    nHidden = net.GetNumHidden()
    for layer in range(0, nHidden + 1):
        if layer == nHidden:
            idxList = net.GetOutputNodeList()
        else:
            idxList = net.GetHiddenLayerNodeList(layer)
        for idx in idxList:
            node = net.GetNode(idx)
            dW = self.speed * delta[idx] * numpy.take(resVect, node.GetInputs())
            newWeights = node.GetWeights() + dW
            node.SetWeights(newWeights)
    return numpy.sqrt(errVect * errVect)[0]