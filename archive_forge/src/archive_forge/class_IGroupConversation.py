from zope.interface import Attribute, Interface
class IGroupConversation(Interface):

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

    def showGroupMessage(sender, text, metadata):
        pass

    def setGroupMembers(members):
        """
        Sets the list of members in the group and displays it to the user.
        """

    def setTopic(topic, author):
        """
        Displays the topic (from the server) for the group conversation window.

        @type topic: string
        @type author: string (XXX: Not Person?)
        """

    def memberJoined(member):
        """
        Adds the given member to the list of members in the group conversation
        and displays this to the user,

        @type member: string (XXX: Not Person?)
        """

    def memberChangedNick(oldnick, newnick):
        """
        Changes the oldnick in the list of members to C{newnick} and displays this
        change to the user,

        @type oldnick: string (XXX: Not Person?)
        @type newnick: string
        """

    def memberLeft(member):
        """
        Deletes the given member from the list of members in the group
        conversation and displays the change to the user.

        @type member: string (XXX: Not Person?)
        """