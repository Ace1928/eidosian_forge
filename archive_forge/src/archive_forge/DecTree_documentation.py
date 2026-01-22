from rdkit.ML.DecTree import Tree
 Constructs and adds a child with the specified data to our list

      **Arguments**

       - name: the name of the new node

       - label: the label of the new node (should be an integer)

       - data: the data to be stored in the new node

       - isTerminal: a toggle to indicate whether or not the new node is
         a terminal (leaf) node.

      **Returns*

        the _DecTreeNode_ which is constructed

    