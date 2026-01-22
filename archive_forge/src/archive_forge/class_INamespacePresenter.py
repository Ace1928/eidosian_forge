from zope.interface import Interface
class INamespacePresenter(Interface):

    def getPersonalNamespaces():
        """
        Report the available personal namespaces.

        Typically there should be only one personal namespace. A common name
        for it is C{""}, and its hierarchical delimiter is usually C{"/"}.

        @rtype: iterable of two-tuples of strings
        @return: The personal namespaces and their hierarchical delimiters. If
            no namespaces of this type exist, None should be returned.
        """

    def getSharedNamespaces():
        """
        Report the available shared namespaces.

        Shared namespaces do not belong to any individual user but are usually
        to one or more of them. Examples of shared namespaces might be
        C{"#news"} for a usenet gateway.

        @rtype: iterable of two-tuples of strings
        @return: The shared namespaces and their hierarchical delimiters. If no
            namespaces of this type exist, None should be returned.
        """

    def getUserNamespaces():
        """
        Report the available user namespaces.

        These are namespaces that contain folders belonging to other users
        access to which this account has been granted.

        @rtype: iterable of two-tuples of strings
        @return: The user namespaces and their hierarchical delimiters. If no
            namespaces of this type exist, None should be returned.
        """