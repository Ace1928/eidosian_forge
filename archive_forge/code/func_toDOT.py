from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from antlr3.tree import CommonTreeAdaptor
from six.moves import range
import stringtemplate3
def toDOT(self, tree, adaptor=None, treeST=_treeST, edgeST=_edgeST):
    if adaptor is None:
        adaptor = CommonTreeAdaptor()
    treeST = treeST.getInstanceOf()
    self.nodeNumber = 0
    self.toDOTDefineNodes(tree, adaptor, treeST)
    self.nodeNumber = 0
    self.toDOTDefineEdges(tree, adaptor, treeST, edgeST)
    return treeST