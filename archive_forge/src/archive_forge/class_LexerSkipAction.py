from enum import IntEnum
class LexerSkipAction(LexerAction):
    INSTANCE = None

    def __init__(self):
        super().__init__(LexerActionType.SKIP)

    def execute(self, lexer: Lexer):
        lexer.skip()

    def __str__(self):
        return 'skip'