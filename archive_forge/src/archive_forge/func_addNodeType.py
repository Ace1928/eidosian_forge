from collections import OrderedDict
from .Node import Node
def addNodeType(self, nodeClass, paths, override=False):
    """
        Register a new node type. If the type's name is already in use,
        an exception will be raised (unless override=True).
        
        ============== =========================================================
        **Arguments:**
        
        nodeClass      a subclass of Node (must have typ.nodeName)
        paths          list of tuples specifying the location(s) this 
                       type will appear in the library tree.
        override       if True, overwrite any class having the same name
        ============== =========================================================
        """
    if not isNodeClass(nodeClass):
        raise Exception('Object %s is not a Node subclass' % str(nodeClass))
    name = nodeClass.nodeName
    if not override and name in self.nodeList:
        raise Exception("Node type name '%s' is already registered." % name)
    self.nodeList[name] = nodeClass
    for path in paths:
        root = self.nodeTree
        for n in path:
            if n not in root:
                root[n] = OrderedDict()
            root = root[n]
        root[name] = nodeClass