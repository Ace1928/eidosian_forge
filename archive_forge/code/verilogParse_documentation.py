import time
import pprint
import sys
from pyparsing import Literal, Keyword, Word, OneOrMore, ZeroOrMore, \
import pyparsing

        <UDP>
        ::= primitive <name_of_UDP> ( <name_of_variable> <,<name_of_variable>>* ) ;
                <UDP_declaration>+
                <UDP_initial_statement>?
                <table_definition>
                endprimitive
        