import sys
import types
import antlr4
from antlr4 import InputStream, CommonTokenStream, Token
from antlr4.tree.Tree import ParseTree
from antlr4.error.ErrorListener import ErrorListener
from .fugue_sqlParser import fugue_sqlParser
from .fugue_sqlLexer import fugue_sqlLexer

        Called when lexer or parser encountered a syntax error.

        Parameters
        ----------
        input_stream:InputStream
            Reference to the original input stream that this error is from

        offendingSymbol:Token
            If available, denotes the erronous token

        char_index:int
            Character offset of the error within the input stream

        line:int
            Line number of the error

        column:int
            Character offset within the line

        msg:str
            Antlr error message
        