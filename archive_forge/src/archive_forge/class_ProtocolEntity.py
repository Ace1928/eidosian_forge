from .protocoltreenode import ProtocolTreeNode
import unittest, time
class ProtocolEntity(object):
    __ID_GEN = 0

    def __init__(self, tag):
        self.tag = tag

    def getTag(self):
        return self.tag

    def isType(self, typ):
        return self.tag == typ

    def _createProtocolTreeNode(self, attributes, children=None, data=None):
        return ProtocolTreeNode(self.getTag(), attributes, children, data)

    def _getCurrentTimestamp(self):
        return int(time.time())

    def _generateId(self, short=False):
        ProtocolEntity.__ID_GEN += 1
        return str(ProtocolEntity.__ID_GEN) if short else str(int(time.time())) + '-' + str(ProtocolEntity.__ID_GEN)

    def toProtocolTreeNode(self):
        pass

    @staticmethod
    def fromProtocolTreeNode(self, protocolTreeNode):
        pass