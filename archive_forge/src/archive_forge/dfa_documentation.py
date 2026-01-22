from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from antlr3.constants import EOF
from antlr3.exceptions import NoViableAltException, BacktrackingFailed
from six.moves import range
@brief Unpack the runlength encoded table data.

        Terence implemented packed table initializers, because Java has a
        size restriction on .class files and the lookup tables can grow
        pretty large. The generated JavaLexer.java of the Java.g example
        would be about 15MB with uncompressed array initializers.

        Python does not have any size restrictions, but the compilation of
        such large source files seems to be pretty memory hungry. The memory
        consumption of the python process grew to >1.5GB when importing a
        15MB lexer, eating all my swap space and I was to impacient to see,
        if it could finish at all. With packed initializers that are unpacked
        at import time of the lexer module, everything works like a charm.

        