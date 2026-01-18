from pyparsing import Word, alphanums, Keyword, Group, Combine, Forward, Suppress, OneOrMore, oneOf
Evaluate quoted strings

        First is does an 'and' on the indidual search terms, then it asks the
        function GetQuoted to only return the subset of ID's that contain the
        literal string.
        