from enum import Enum
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNConfig import ATNConfig
from antlr4.atn.ATNConfigSet import ATNConfigSet
from antlr4.atn.ATNState import RuleStopState
from antlr4.atn.SemanticContext import SemanticContext
@classmethod
def allConfigsInRuleStopStates(cls, configs: ATNConfigSet):
    return all((isinstance(cfg.state, RuleStopState) for cfg in configs))