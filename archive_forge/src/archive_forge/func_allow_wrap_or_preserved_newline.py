import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
def allow_wrap_or_preserved_newline(self, current_token, force_linewrap=False):
    if self._output.just_added_newline():
        return
    shouldPreserveOrForce = self._options.preserve_newlines and current_token.newlines or force_linewrap
    operatorLogicApplies = self._flags.last_token.text in Tokenizer.positionable_operators or current_token.text in Tokenizer.positionable_operators
    if operatorLogicApplies:
        shouldPrintOperatorNewline = self._flags.last_token.text in Tokenizer.positionable_operators and self._options.operator_position in OPERATOR_POSITION_BEFORE_OR_PRESERVE or current_token.text in Tokenizer.positionable_operators
        shouldPreserveOrForce = shouldPreserveOrForce and shouldPrintOperatorNewline
    if shouldPreserveOrForce:
        self.print_newline(preserve_statement_flags=True)
    elif self._options.wrap_line_length > 0:
        if reserved_array(self._flags.last_token, self._newline_restricted_tokens):
            return
        self._output.set_wrap_point()