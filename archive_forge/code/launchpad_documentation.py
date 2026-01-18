from datetime import datetime
import sys
Look up a slice, or a subordinate resource by index.

        @param key: An individual object key or a C{slice}.
        @raises IndexError: Raised if an invalid key is provided.
        @return: A L{FakeResource} instance for the entry matching C{key}.
        