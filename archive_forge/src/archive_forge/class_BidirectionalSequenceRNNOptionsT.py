import flatbuffers
from flatbuffers.compat import import_numpy
class BidirectionalSequenceRNNOptionsT(object):

    def __init__(self):
        self.timeMajor = False
        self.fusedActivationFunction = 0
        self.mergeOutputs = False
        self.asymmetricQuantizeInputs = False

    @classmethod
    def InitFromBuf(cls, buf, pos):
        bidirectionalSequenceRnnoptions = BidirectionalSequenceRNNOptions()
        bidirectionalSequenceRnnoptions.Init(buf, pos)
        return cls.InitFromObj(bidirectionalSequenceRnnoptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, bidirectionalSequenceRnnoptions):
        x = BidirectionalSequenceRNNOptionsT()
        x._UnPack(bidirectionalSequenceRnnoptions)
        return x

    def _UnPack(self, bidirectionalSequenceRnnoptions):
        if bidirectionalSequenceRnnoptions is None:
            return
        self.timeMajor = bidirectionalSequenceRnnoptions.TimeMajor()
        self.fusedActivationFunction = bidirectionalSequenceRnnoptions.FusedActivationFunction()
        self.mergeOutputs = bidirectionalSequenceRnnoptions.MergeOutputs()
        self.asymmetricQuantizeInputs = bidirectionalSequenceRnnoptions.AsymmetricQuantizeInputs()

    def Pack(self, builder):
        BidirectionalSequenceRNNOptionsStart(builder)
        BidirectionalSequenceRNNOptionsAddTimeMajor(builder, self.timeMajor)
        BidirectionalSequenceRNNOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        BidirectionalSequenceRNNOptionsAddMergeOutputs(builder, self.mergeOutputs)
        BidirectionalSequenceRNNOptionsAddAsymmetricQuantizeInputs(builder, self.asymmetricQuantizeInputs)
        bidirectionalSequenceRnnoptions = BidirectionalSequenceRNNOptionsEnd(builder)
        return bidirectionalSequenceRnnoptions