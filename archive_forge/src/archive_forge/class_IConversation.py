from zope.interface import Attribute, Interface
class IConversation(Interface):
    """
    A conversation with a specific person.
    """

    def __init__(person, chatui):
        """
        @type person: L{IPerson}
        """

    def show():
        """
        doesn't seem like it belongs in this interface.
        """

    def hide():
        """
        nor this neither.
        """

    def sendText(text, metadata):
        pass

    def showMessage(text, metadata):
        pass

    def changedNick(person, newnick):
        """
        @param person: XXX Shouldn't this always be Conversation.person?
        """