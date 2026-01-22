from pyparsing import *
import_stmt: import_name | import_from
import_name: 'import' dotted_as_names
import_from: 'from' dotted_name 'import' ('*' | '(' import_as_names ')' | import_as_names)
import_as_name: NAME [NAME NAME]
import_as_names: import_as_name (',' import_as_name)* [',']
class OptionalGroup(SemanticGroup):
    label = 'OPT'
    pass