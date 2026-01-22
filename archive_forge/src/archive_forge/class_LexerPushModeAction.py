from enum import IntEnum
class LexerPushModeAction(LexerAction):
    __slots__ = 'mode'

    def __init__(self, mode: int):
        super().__init__(LexerActionType.PUSH_MODE)
        self.mode = mode

    def execute(self, lexer: Lexer):
        lexer.pushMode(self.mode)

    def __hash__(self):
        return hash((self.actionType, self.mode))

    def __eq__(self, other):
        if self is other:
            return True
        elif not isinstance(other, LexerPushModeAction):
            return False
        else:
            return self.mode == other.mode

    def __str__(self):
        return 'pushMode(' + str(self.mode) + ')'