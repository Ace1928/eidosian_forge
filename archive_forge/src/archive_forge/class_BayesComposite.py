import numpy
from rdkit.ML.Composite import Composite
class BayesComposite(Composite.Composite):
    """a composite model using Bayesian statistics in the Decision Proxy


    **Notes**

    - typical usage:

       1) grow the composite with AddModel until happy with it

       2) call AverageErrors to calculate the average error values

       3) call SortModels to put things in order by either error or count

       4) call Train to update the Bayesian stats.

  """

    def Train(self, data, verbose=0):
        nModels = len(self)
        nResults = self.nPossibleVals[-1]
        self.resultProbs = numpy.zeros(nResults, float)
        self.condProbs = [None] * nModels
        for i in range(nModels):
            self.condProbs[i] = numpy.zeros((nResults, nResults), float)
        for example in data:
            act = self.QuantizeActivity(example)[-1]
            self.resultProbs[int(act)] += 1
        for example in data:
            if self._mapOrder is not None:
                example = self._RemapInput(example)
            if self.GetActivityQuantBounds():
                example = self.QuantizeActivity(example)
            if self.quantBounds is not None and 1 in self.quantizationRequirements:
                quantExample = self.QuantizeExample(example, self.quantBounds)
            else:
                quantExample = []
            trueRes = int(example[-1])
            votes = self.CollectVotes(example, quantExample)
            for i in range(nModels):
                self.condProbs[i][votes[i], trueRes] += 1
        for i in range(nModels):
            for j in range(nResults):
                self.condProbs[i][j] /= sum(self.condProbs[i][j])
        self.resultProbs /= sum(self.resultProbs)
        if verbose:
            print('**** Bayesian Results')
            print('Result probabilities')
            print('\t', self.resultProbs)
            print('Model by model breakdown of conditional probs')
            for mat in self.condProbs:
                for row in mat:
                    print('\t', row)
                print()

    def ClassifyExample(self, example, threshold=0, verbose=0, appendExample=0):
        """ classifies the given example using the entire composite

      **Arguments**

       - example: the data to be classified

       - threshold:  if this is a number greater than zero, then a
          classification will only be returned if the confidence is
          above _threshold_.  Anything lower is returned as -1.

      **Returns**

        a (result,confidence) tuple

    """
        if self._mapOrder is not None:
            example = self._RemapInput(example)
        if self.GetActivityQuantBounds():
            example = self.QuantizeActivity(example)
        if self.quantBounds is not None and 1 in self.quantizationRequirements:
            quantExample = self.QuantizeExample(example, self.quantBounds)
        else:
            quantExample = []
        self.modelVotes = self.CollectVotes(example, quantExample, appendExample=appendExample)
        nPossibleRes = self.nPossibleVals[-1]
        votes = [0.0] * nPossibleRes
        for i in range(len(self)):
            predict = self.modelVotes[i]
            for j in range(nPossibleRes):
                votes[j] += self.condProbs[i][predict, j]
        res = numpy.argmax(votes)
        conf = votes[res] / len(self)
        if verbose:
            print(votes, conf, example[-1])
        if conf > threshold:
            return (res, conf)
        else:
            return (-1, conf)

    def __init__(self):
        Composite.Composite.__init__(self)
        self.resultProbs = None
        self.condProbs = None