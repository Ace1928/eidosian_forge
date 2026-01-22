from zope.interface import Attribute, Interface
class IInitiatingInitializer(IInitializer):
    """
    Interface for XML stream initializers for the initiating entity.
    """
    xmlstream = Attribute('The associated XML stream')

    def initialize():
        """
        Initiate the initialization step.

        May return a deferred when the initialization is done asynchronously.
        """