import os
import coverage
from kivy.lang.parser import Parser
def walk_parser_rules(parser_rule):
    yield parser_rule
    for child in parser_rule.children:
        for rule in walk_parser_rules(child):
            yield rule
    if parser_rule.canvas_before is not None:
        for rule in walk_parser_rules(parser_rule.canvas_before):
            yield rule
        yield parser_rule.canvas_before
    if parser_rule.canvas_root is not None:
        for rule in walk_parser_rules(parser_rule.canvas_root):
            yield rule
    if parser_rule.canvas_after is not None:
        for rule in walk_parser_rules(parser_rule.canvas_after):
            yield rule