import unittest
import inspect
import threading
class YowParallelLayer(YowLayer):

    def __init__(self, sublayers=None):
        super(YowParallelLayer, self).__init__()
        self.sublayers = sublayers or []
        self.sublayers = tuple([sublayer() for sublayer in sublayers])
        for s in self.sublayers:
            s.toLower = self.toLower
            s.toUpper = self.toUpper
            s.broadcastEvent = self.subBroadcastEvent
            s.emitEvent = self.subEmitEvent

    def getLayerInterface(self, YowLayerClass):
        for s in self.sublayers:
            if s.__class__ == YowLayerClass:
                return s.getLayerInterface()

    def setStack(self, stack):
        super(YowParallelLayer, self).setStack(stack)
        for s in self.sublayers:
            s.setStack(self.getStack())

    def receive(self, data):
        for s in self.sublayers:
            s.receive(data)

    def send(self, data):
        for s in self.sublayers:
            s.send(data)

    def subBroadcastEvent(self, yowLayerEvent):
        self.onEvent(yowLayerEvent)
        self.broadcastEvent(yowLayerEvent)

    def subEmitEvent(self, yowLayerEvent):
        self.onEvent(yowLayerEvent)
        self.emitEvent(yowLayerEvent)

    def onEvent(self, yowLayerEvent):
        stopEvent = False
        for s in self.sublayers:
            stopEvent = stopEvent or s.onEvent(yowLayerEvent)
        return stopEvent

    def __str__(self):
        return ' - '.join([l.__str__() for l in self.sublayers])