from datetime import datetime
import sys
def _run_method(self, name, method, *args, **kwargs):
    """Run a method and convert its result into a L{FakeResource}.

        If the result represents an object it is validated against the WADL
        definition before being returned.

        @param name: The name of the method.
        @param method: A callable.
        @param args: Arguments to pass to the callable.
        @param kwargs: Keyword arguments to pass to the callable.
        @return: A L{FakeResource} representing the result if it's an object.
        @raises IntegrityError: Raised if the return value from the method
            isn't valid.
        """
    result = method(*args, **kwargs)
    if name in self.special_methods or result is None:
        return result
    else:
        return self._create_resource(self._resource_type, name, result)