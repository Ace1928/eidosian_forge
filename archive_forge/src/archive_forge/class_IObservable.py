import abc
class IObservable(abc.ABC):
    """ Interface for objects that can emit notifications for the observer
    system.
    """

    def _notifiers(self, force_create):
        """ Return a list of callables where each callable is a notifier.
        The list is expected to be mutated for contributing or removing
        notifiers from the object.

        Parameters
        ----------
        force_create: boolean
            It is added for compatibility with CTrait.
            It should not be used otherwise.
        """
        raise NotImplementedError('Observable object must implement _notifiers')