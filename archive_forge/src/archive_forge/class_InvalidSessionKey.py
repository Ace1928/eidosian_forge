from django.core.exceptions import BadRequest, SuspiciousOperation
class InvalidSessionKey(SuspiciousOperation):
    """Invalid characters in session key"""
    pass