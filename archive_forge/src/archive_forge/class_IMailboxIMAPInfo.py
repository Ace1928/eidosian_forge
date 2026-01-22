from zope.interface import Interface
class IMailboxIMAPInfo(Interface):
    """
    Interface specifying only the methods required for C{listMailboxes}.

    Implementations can return objects implementing only these methods for
    return to C{listMailboxes} if it can allow them to operate more
    efficiently.
    """

    def getFlags():
        """
        Return the flags defined in this mailbox

        Flags with the \\ prefix are reserved for use as system flags.

        @rtype: L{list} of L{str}
        @return: A list of the flags that can be set on messages in this
            mailbox.
        """

    def getHierarchicalDelimiter():
        """
        Get the character which delimits namespaces for in this mailbox.

        @rtype: L{bytes}
        """