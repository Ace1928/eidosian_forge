from enum import IntEnum
class LexerMoreAction(LexerAction):
    INSTANCE = None

    def __init__(self):
        super().__init__(LexerActionType.MORE)

    def execute(self, lexer: Lexer):
        lexer.more()

    def __str__(self):
        return 'more'