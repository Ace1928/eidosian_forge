from functools import wraps
from django.utils.functional import keep_lazy

        Concatenating a safe string with another safe bytestring or
        safe string is safe. Otherwise, the result is no longer safe.
        