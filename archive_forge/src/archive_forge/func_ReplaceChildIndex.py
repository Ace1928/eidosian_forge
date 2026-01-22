import pickle
def ReplaceChildIndex(self, index, newChild):
    """ Replaces a given child with a new one

      **Arguments**

        - index: an integer

        - child: a TreeNode

    """
    self.children[index] = newChild