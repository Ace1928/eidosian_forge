from __future__ import annotations
from .state_inline import Delimiter, StateInline
def _postProcess(state: StateInline, delimiters: list[Delimiter]) -> None:
    loneMarkers = []
    maximum = len(delimiters)
    i = 0
    while i < maximum:
        startDelim = delimiters[i]
        if startDelim.marker != 126:
            i += 1
            continue
        if startDelim.end == -1:
            i += 1
            continue
        endDelim = delimiters[startDelim.end]
        token = state.tokens[startDelim.token]
        token.type = 's_open'
        token.tag = 's'
        token.nesting = 1
        token.markup = '~~'
        token.content = ''
        token = state.tokens[endDelim.token]
        token.type = 's_close'
        token.tag = 's'
        token.nesting = -1
        token.markup = '~~'
        token.content = ''
        if state.tokens[endDelim.token - 1].type == 'text' and state.tokens[endDelim.token - 1].content == '~':
            loneMarkers.append(endDelim.token - 1)
        i += 1
    while loneMarkers:
        i = loneMarkers.pop()
        j = i + 1
        while j < len(state.tokens) and state.tokens[j].type == 's_close':
            j += 1
        j -= 1
        if i != j:
            token = state.tokens[j]
            state.tokens[j] = state.tokens[i]
            state.tokens[i] = token