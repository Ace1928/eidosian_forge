from zope.interface import Attribute, Interface
def unregisterAccountClient(client):
    """
        Notifies user that an account has been signed off or disconnected.

        @type client: L{Client<IClient>}
        """