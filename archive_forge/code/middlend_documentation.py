from pythran.analyses import ExtendedSyntaxCheck
from pythran.optimizations import (ComprehensionPatterns, ListCompToGenexp,
from pythran.transformations import (ExpandBuiltins, ExpandImports,
import re
 Refine node in place until it matches pythran's expectations. 