from .low_level import Message, MessageType, HeaderFields
from .wrappers import MessageGenerator, new_method_call
def RemoveMatch(self, rule):
    if isinstance(rule, MatchRule):
        rule = rule.serialise()
    return new_method_call(self, 'RemoveMatch', 's', (rule,))