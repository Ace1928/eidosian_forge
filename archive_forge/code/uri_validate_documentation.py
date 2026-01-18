from __future__ import unicode_literals
import re
Regex for URIs

These regex are directly derived from the collected ABNF in RFC3986
(except for DIGIT, ALPHA and HEXDIG, defined by RFC2234).

They should be processed with re.VERBOSE.

Thanks Mark Nottingham for this code - https://gist.github.com/138549
