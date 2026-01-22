import gast as ast
import os
import re
from time import time
class Transformation(ContextManager, ast.NodeTransformer):
    """A pass that updates its content."""

    def __init__(self, *args, **kwargs):
        """ Initialize the update used to know if update happened. """
        super(Transformation, self).__init__(*args, **kwargs)
        self.update = False

    def run(self, node):
        """ Apply transformation and dependencies and fix new node location."""
        n = super(Transformation, self).run(node)
        if self.update:
            self.passmanager._cache.clear()
        return n

    def apply(self, node):
        """ Apply transformation and return if an update happened. """
        new_node = self.run(node)
        return (self.update, new_node)