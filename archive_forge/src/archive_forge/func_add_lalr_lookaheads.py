import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def add_lalr_lookaheads(self, C):
    nullable = self.compute_nullable_nonterminals()
    trans = self.find_nonterminal_transitions(C)
    readsets = self.compute_read_sets(C, trans, nullable)
    lookd, included = self.compute_lookback_includes(C, trans, nullable)
    followsets = self.compute_follow_sets(trans, readsets, included)
    self.add_lookaheads(lookd, followsets)