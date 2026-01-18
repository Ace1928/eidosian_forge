import logging
def universal_quantifier(predicate, domain):
    return all((predicate(x) for x in domain))