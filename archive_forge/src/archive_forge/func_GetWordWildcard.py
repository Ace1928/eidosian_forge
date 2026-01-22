from searchparser import SearchQueryParser
def GetWordWildcard(self, word):
    return {p for p in products if p.startswith(word[:-1])}