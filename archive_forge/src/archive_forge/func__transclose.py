import os
import re
import shelve
import sys
import nltk.data
def _transclose(self, g):
    """
        Compute the transitive closure of a graph represented as a linked list.
        """
    for x in g:
        for adjacent in g[x]:
            if adjacent in g:
                for y in g[adjacent]:
                    if y not in g[x]:
                        g[x].append(y)
    return g