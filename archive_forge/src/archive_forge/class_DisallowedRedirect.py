import operator
from django.utils.hashable import make_hashable
class DisallowedRedirect(SuspiciousOperation):
    """Redirect to scheme not in allowed list"""
    pass