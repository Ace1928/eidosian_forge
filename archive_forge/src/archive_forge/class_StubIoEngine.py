from modin.config import Engine, IsExperimental, StorageFormat
from modin.core.execution.dispatching.factories import factories
from modin.utils import _inherit_docstrings, get_current_execution
class StubIoEngine(object):
    """
    IO-Engine that does nothing more than raise NotImplementedError when any method is called.

    Parameters
    ----------
    factory_name : str
        Factory name, which will be reflected in error messages.

    Notes
    -----
    Used for testing purposes.
    """

    def __init__(self, factory_name=''):
        self.factory_name = factory_name or 'Unknown'

    def __getattr__(self, name):
        """
        Return a function that raises `NotImplementedError` for the `name` method.

        Parameters
        ----------
        name : str
            Method name to indicate in `NotImplementedError`.

        Returns
        -------
        callable
        """

        def stub(*args, **kw):
            raise NotImplementedError(f'Method {self.factory_name}.{name} is not implemented')
        return stub