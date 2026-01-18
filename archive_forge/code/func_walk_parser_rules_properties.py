import os
import coverage
from kivy.lang.parser import Parser
def walk_parser_rules_properties(parser_rule):
    for rule in parser_rule.properties.values():
        yield rule
    for rule in parser_rule.handlers:
        yield rule