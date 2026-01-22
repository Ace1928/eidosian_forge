import pickle
def NameTree(self, varNames):
    """ Set the names of each node in the tree from a list of variable names.

     **Arguments**

       - varNames: a list of names to be assigned

     **Notes**

        1) this works its magic by recursively traversing all children

        2) The assumption is made here that the varNames list can be indexed
           by the labels of tree nodes

    """
    if self.GetTerminal():
        return
    else:
        for child in self.GetChildren():
            child.NameTree(varNames)
        self.SetName(varNames[self.GetLabel()])