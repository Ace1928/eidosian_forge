import os
import re
import shelve
import sys
import nltk.data

        Close a binary relation in the ``Concept``'s extension set.

        :return: a new extension for the ``Concept`` in which the
                 relation is closed under a given property
        