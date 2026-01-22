import re
from urllib import parse as urllib_parse
import pyparsing as pp
def _parsed_query2dict(parsed_query):
    result = None
    while parsed_query:
        part = parsed_query.pop()
        if part in binary_operator:
            result = {part: {parsed_query.pop(): result}}
        elif part in multiple_operators:
            if result.get(part):
                result[part].append(_parsed_query2dict(parsed_query.pop()))
            else:
                result = {part: [result]}
        elif part in uninary_operators:
            result = {part: result}
        elif isinstance(part, pp.ParseResults):
            kind = part.getName()
            if kind == 'list':
                res = part.asList()
            else:
                res = _parsed_query2dict(part)
            if result is None:
                result = res
            elif isinstance(result, dict):
                list(result.values())[0].append(res)
        else:
            result = part
    return result