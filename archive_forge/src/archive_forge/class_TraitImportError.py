from .trait_errors import TraitError
class TraitImportError(TraitFactory):
    """ TraitFactory subclass that always fails when creating a CTrait

    This class is designed for uses such as deferring import problems until
    encountering code that actually tries to use the unimportable trait.
    """

    def __init__(self, message):
        self.message = message

    def __call__(self, *args, **metadata):
        """ Raises an TraitError with the message as is payload. """
        raise TraitError(self.message)