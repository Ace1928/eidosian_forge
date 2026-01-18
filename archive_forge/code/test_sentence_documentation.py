import itertools
from twisted.positioning import _sentence
from twisted.trial.unittest import TestCase

        L{None} does not appear in the sentence attributes of the
        protocol, even though it's in the specification.

        This is because L{None} is a placeholder for parts of the sentence you
        don't really need or want, but there are some bits later on in the
        sentence that you do want. The alternative would be to have to specify
        things like "_UNUSED0", "_UNUSED1"... which would end up cluttering
        the sentence data and eventually adapter state.
        