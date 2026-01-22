import os.path
import sys
import codecs
import argparse
from lark import Lark, Transformer, v_args
def _extra_rule(self, rule):
    if rule in self.extra_rules_rev:
        return self.extra_rules_rev[rule]
    name = 'xrule_%d' % len(self.extra_rules)
    assert name not in self.extra_rules
    self.extra_rules[name] = rule
    self.extra_rules_rev[rule] = name
    return name