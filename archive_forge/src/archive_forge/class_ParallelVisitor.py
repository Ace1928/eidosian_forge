from copy import copy
from . import ast
from .visitor_meta import QUERY_DOCUMENT_KEYS, VisitorMeta
class ParallelVisitor(Visitor):
    __slots__ = ('skipping', 'visitors')

    def __init__(self, visitors):
        self.visitors = visitors
        self.skipping = [None] * len(visitors)

    def enter(self, node, key, parent, path, ancestors):
        for i, visitor in enumerate(self.visitors):
            if not self.skipping[i]:
                result = visitor.enter(node, key, parent, path, ancestors)
                if result is False:
                    self.skipping[i] = node
                elif result is BREAK:
                    self.skipping[i] = BREAK
                elif result is not None:
                    return result

    def leave(self, node, key, parent, path, ancestors):
        for i, visitor in enumerate(self.visitors):
            if not self.skipping[i]:
                result = visitor.leave(node, key, parent, path, ancestors)
                if result is BREAK:
                    self.skipping[i] = BREAK
                elif result is not None and result is not False:
                    return result
            elif self.skipping[i] == node:
                self.skipping[i] = REMOVE