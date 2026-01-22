from typing import Set
class _PositioningSentenceProducerMixin:
    """
    A mixin for certain protocols that produce positioning sentences.

    This mixin helps protocols that store the layout of sentences that they
    consume in a C{_SENTENCE_CONTENTS} class variable provide all sentence
    attributes that can ever occur. It does this by providing a class method,
    C{getSentenceAttributes}, which iterates over all sentence types and
    collects the possible sentence attributes.
    """

    @classmethod
    def getSentenceAttributes(cls):
        """
        Returns a set of all attributes that might be found in the sentences
        produced by this protocol.

        This is basically a set of all the attributes of all the sentences that
        this protocol can produce.

        @return: The set of all possible sentence attribute names.
        @rtype: C{set} of C{str}
        """
        attributes = {'type'}
        for attributeList in cls._SENTENCE_CONTENTS.values():
            for attribute in attributeList:
                if attribute is None:
                    continue
                attributes.add(attribute)
        return attributes