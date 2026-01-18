from ..Qt import QtCore, QtGui, QtWidgets
def updateFlags(self):
    opts = self.param.opts
    flags = QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled
    if opts.get('renamable', False):
        if opts.get('title', None) is not None:
            raise Exception('Cannot make parameter with both title != None and renamable == True.')
        flags |= QtCore.Qt.ItemFlag.ItemIsEditable
    if opts.get('movable', False):
        flags |= QtCore.Qt.ItemFlag.ItemIsDragEnabled
    if opts.get('dropEnabled', False):
        flags |= QtCore.Qt.ItemFlag.ItemIsDropEnabled
    self.setFlags(flags)