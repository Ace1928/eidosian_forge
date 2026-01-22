def AddChildren(self, children):
    """Adds a bunch of children to our list

          **Arguments**

            - children: a list of Clusters

        """
    self.children += children
    self._GenPoints()
    self._UpdateLength()