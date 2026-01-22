from __future__ import (absolute_import, division, print_function)
import re
def _identifier_parse(identifier, quote_char):
    if not identifier:
        raise SQLParseError('Identifier name unspecified or unquoted trailing dot')
    already_quoted = False
    if identifier.startswith(quote_char):
        already_quoted = True
        try:
            end_quote = _find_end_quote(identifier[1:], quote_char=quote_char) + 1
        except UnclosedQuoteError:
            already_quoted = False
        else:
            if end_quote < len(identifier) - 1:
                if identifier[end_quote + 1] == '.':
                    dot = end_quote + 1
                    first_identifier = identifier[:dot]
                    next_identifier = identifier[dot + 1:]
                    further_identifiers = _identifier_parse(next_identifier, quote_char)
                    further_identifiers.insert(0, first_identifier)
                else:
                    raise SQLParseError('User escaped identifiers must escape extra quotes')
            else:
                further_identifiers = [identifier]
    if not already_quoted:
        try:
            dot = identifier.index('.')
        except ValueError:
            identifier = identifier.replace(quote_char, quote_char * 2)
            identifier = ''.join((quote_char, identifier, quote_char))
            further_identifiers = [identifier]
        else:
            if dot == 0 or dot >= len(identifier) - 1:
                identifier = identifier.replace(quote_char, quote_char * 2)
                identifier = ''.join((quote_char, identifier, quote_char))
                further_identifiers = [identifier]
            else:
                first_identifier = identifier[:dot]
                next_identifier = identifier[dot + 1:]
                further_identifiers = _identifier_parse(next_identifier, quote_char)
                first_identifier = first_identifier.replace(quote_char, quote_char * 2)
                first_identifier = ''.join((quote_char, first_identifier, quote_char))
                further_identifiers.insert(0, first_identifier)
    return further_identifiers