from .errors import BzrError, InternalBzrError
Take a list of imports, and split it into regularized form.

        This is meant to take regular import text, and convert it to
        the forms that the rest of the converters prefer.
        