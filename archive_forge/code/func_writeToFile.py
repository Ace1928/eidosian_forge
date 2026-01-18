from .iq_picture import PictureIqProtocolEntity
from yowsup.structs import ProtocolTreeNode
def writeToFile(self, path):
    with open(path, 'wb') as outFile:
        outFile.write(self.getPictureData())