import os
import coverage
from kivy.lang.parser import Parser
def walk_parser(parser):
    if parser.root is not None:
        for rule in walk_parser_rules(parser.root):
            for prop in walk_parser_rules_properties(rule):
                yield prop
    for _, cls_rule in parser.rules:
        for rule in walk_parser_rules(cls_rule):
            for prop in walk_parser_rules_properties(rule):
                yield prop